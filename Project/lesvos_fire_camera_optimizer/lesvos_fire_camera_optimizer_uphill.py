# -*- coding: utf-8 -*-
"""
QGIS Processing Script: Lesvos Fire Camera Optimizer

Finds a greedy near-optimal set of camera locations and azimuths using:
- a projected DEM (metres),
- an aligned ESA WorldCover raster,
- GDAL viewshed,
- a horizontal field of view (default 180 degrees),
- land-cover risk weights,
- distance-dependent detection quality,
- local uphill refinement around each initially selected camera.

The algorithm first finds coarse/global camera positions and then searches nearby
higher terrain. A refined position is accepted only if the overall weighted
coverage remains within the user-defined allowed loss.

Designed for QGIS 3.34+ with the GDAL Processing provider enabled.
"""

import csv
import math
import os
import shutil
import tempfile

import numpy as np
from osgeo import gdal

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsApplication,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingUtils,
    QgsWkbTypes,
)
from qgis import processing


gdal.UseExceptions()


class LesvosFireCameraOptimizer(QgsProcessingAlgorithm):
    DEM = "DEM"
    LANDCOVER = "LANDCOVER"
    CANDIDATES = "CANDIDATES"
    NUM_CAMERAS = "NUM_CAMERAS"
    FOV = "FOV"
    AZ_STEP = "AZ_STEP"
    OBSERVER_HEIGHT = "OBSERVER_HEIGHT"
    TARGET_HEIGHT = "TARGET_HEIGHT"
    MAX_DISTANCE = "MAX_DISTANCE"
    GRID_SPACING = "GRID_SPACING"
    MAX_CANDIDATES = "MAX_CANDIDATES"
    MIN_ELEVATION = "MIN_ELEVATION"
    REFINE_RADIUS = "REFINE_RADIUS"
    REFINE_STEP = "REFINE_STEP"
    REFINE_MAX_LOSS = "REFINE_MAX_LOSS"
    REFINE_MAX_CANDIDATES = "REFINE_MAX_CANDIDATES"

    OUT_INITIAL_CAMERAS = "OUT_INITIAL_CAMERAS"
    OUT_CAMERAS = "OUT_CAMERAS"
    OUT_MOVES = "OUT_MOVES"
    OUT_FOV = "OUT_FOV"
    OUT_QUALITY = "OUT_QUALITY"
    OUT_COUNT = "OUT_COUNT"
    OUT_REMAINING_RISK = "OUT_REMAINING_RISK"
    OUT_SUMMARY = "OUT_SUMMARY"

    # ESA WorldCover 2021 codes -> relative fire-monitoring importance.
    # These are modelling assumptions, not official fire-risk coefficients.
    WORLDCOVER_WEIGHTS = {
        10: 25.0,   # Δενδρώδης κάλυψη / δασικές εκτάσεις
        20: 18.0,   # Θαμνώδεις εκτάσεις
        30: 10.0,    # Χορτολιβαδικές εκτάσεις
        40: 3.0,    # Καλλιεργούμενες εκτάσεις
        50: 0.0,    # Δομημένες περιοχές
        60: 0.0,    # Γυμνό έδαφος / αραιή βλάστηση
        70: 0.0,    # Χιόνι / πάγος
        80: 0.0,    # Μόνιμα ύδατα
        90: 2.0,    # Ποώδεις υγρότοποι
        95: 0.0,    # Μαγκρόβια
        100: 0.0,   # Βρύα / λειχήνες
    }

    def tr(self, text):
        return QCoreApplication.translate("LesvosFireCameraOptimizer", text)

    def createInstance(self):
        return LesvosFireCameraOptimizer()

    def name(self):
        # Unique id so QGIS does not confuse this version with an older cached script.
        return "lesvos_fire_camera_optimizer_uphill_v2"

    def displayName(self):
        return self.tr("Βελτιστοποίηση θέσεων καμερών – uphill refinement")

    def group(self):
        return self.tr("Πυρανίχνευση")

    def groupId(self):
        return "fire_detection"

    def shortHelpString(self):
        return self.tr(
            "Επιλέγει θέσεις και κατευθύνσεις καμερών με βάση DEM, WorldCover, "
            "viewshed, οριζόντιο FOV και απόσταση. Για πρώτη δοκιμή στη Λέσβο: "
            "FOV=180°, ύψος κάμερας=10 m, ύψος στόχου=20 m, εμβέλεια=4000 m, "
            "βήμα υποψήφιων θέσεων=3000 m, βήμα azimuth=30°. Μετά την αρχική "
            "greedy επιλογή κάνει τοπική αναζήτηση σε υψηλότερο έδαφος γύρω από "
            "κάθε κάμερα, κρατώντας τη συνολική κάλυψη μέσα στο επιτρεπόμενο όριο απώλειας. "
            "Τα δύο raster πρέπει να είναι ευθυγραμμισμένα, στο ίδιο projected CRS και σε μέτρα."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.DEM,
                self.tr("DEM υψομέτρου (projected CRS σε μέτρα)"),
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.LANDCOVER,
                self.tr("ESA WorldCover aligned με το DEM"),
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.CANDIDATES,
                self.tr("Προαιρετικά: έτοιμα υποψήφια σημεία καμερών"),
                [QgsProcessing.TypeVectorPoint],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.NUM_CAMERAS,
                self.tr("Αριθμός καμερών που θα επιλεγούν"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=1,
                maxValue=100,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.FOV,
                self.tr("Οριζόντιο FOV κάθε κάμερας (μοίρες)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=180.0,
                minValue=1.0,
                maxValue=360.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.AZ_STEP,
                self.tr("Βήμα δοκιμής azimuth (μοίρες)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=5.0,
                minValue=1.0,
                maxValue=180.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OBSERVER_HEIGHT,
                self.tr("Ύψος κάμερας πάνω από το έδαφος (m)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=4.0,
                minValue=0.1,
                maxValue=200.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TARGET_HEIGHT,
                self.tr("Ύψος στόχου/καπνού πάνω από το έδαφος (m)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=20.0,
                minValue=0.0,
                maxValue=1000.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_DISTANCE,
                self.tr("Μέγιστη απόσταση ανίχνευσης (m)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=4000.0,
                minValue=100.0,
                maxValue=100000.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.GRID_SPACING,
                self.tr("Απόσταση αυτόματων υποψήφιων θέσεων (m)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=600.0,
                minValue=50.0,
                maxValue=20000.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_CANDIDATES,
                self.tr("Μέγιστος αριθμός υποψήφιων θέσεων"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5000,
                minValue=1,
                maxValue=5000,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_ELEVATION,
                self.tr("Ελάχιστο υψόμετρο υποψήφιας θέσης (m)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=-500.0,
                maxValue=5000.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.REFINE_RADIUS,
                self.tr("Ακτίνα τοπικής uphill αναζήτησης (m)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=750.0,
                minValue=0.0,
                maxValue=5000.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.REFINE_STEP,
                self.tr("Βήμα τοπικής αναζήτησης (m)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=90.0,
                minValue=30.0,
                maxValue=1000.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.REFINE_MAX_LOSS,
                self.tr("Μέγιστη επιτρεπόμενη απώλεια συνολικού score στο refinement (%)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0,
                minValue=0.0,
                maxValue=20.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.REFINE_MAX_CANDIDATES,
                self.tr("Μέγιστοι τοπικοί uphill υποψήφιοι ανά κάμερα"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=80,
                minValue=1,
                maxValue=500,
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUT_INITIAL_CAMERAS,
                self.tr("Αρχικές θέσεις καμερών (πριν το uphill refinement)"),
                QgsProcessing.TypeVectorPoint,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUT_CAMERAS,
                self.tr("Τελικές θέσεις καμερών (μετά το uphill refinement)"),
                QgsProcessing.TypeVectorPoint,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUT_MOVES,
                self.tr("Μετακινήσεις αρχική → τελική θέση"),
                QgsProcessing.TypeVectorLine,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUT_FOV,
                self.tr("Γεωμετρικοί τομείς FOV των επιλεγμένων καμερών"),
                QgsProcessing.TypeVectorPolygon,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUT_QUALITY,
                self.tr("Μέγιστη ποιότητα κάλυψης (0–1)"),
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUT_COUNT,
                self.tr("Πλήθος καμερών που καλύπτουν κάθε κελί"),
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUT_REMAINING_RISK,
                self.tr("Υπολειπόμενος σταθμισμένος κίνδυνος"),
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUT_SUMMARY,
                self.tr("Αναφορά αποτελεσμάτων CSV"),
                fileFilter="CSV files (*.csv)",
            )
        )

    @staticmethod
    def _same_grid(ds_a, ds_b):
        if ds_a.RasterXSize != ds_b.RasterXSize or ds_a.RasterYSize != ds_b.RasterYSize:
            return False
        gt_a = ds_a.GetGeoTransform()
        gt_b = ds_b.GetGeoTransform()
        return all(abs(a - b) <= 1e-6 for a, b in zip(gt_a, gt_b))

    @staticmethod
    def _xy_to_row_col(x, y, gt, rows, cols):
        if gt[2] != 0 or gt[4] != 0:
            raise QgsProcessingException(
                "Το script δεν υποστηρίζει περιστραμμένο raster grid. Κάνε Align rasters ξανά."
            )
        col = int(math.floor((x - gt[0]) / gt[1]))
        row = int(math.floor((y - gt[3]) / gt[5]))
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return None
        return row, col

    @staticmethod
    def _cell_centers(rows_idx, cols_idx, gt):
        xs = gt[0] + (cols_idx.astype(np.float64) + 0.5) * gt[1]
        ys = gt[3] + (rows_idx.astype(np.float64) + 0.5) * gt[5]
        return xs, ys

    @staticmethod
    def _distance_quality(distance, max_distance):
        # Smooth detection-quality assumption:
        # q=1 near the camera and q≈0.135 at maximum range.
        ratio = np.clip(distance / max_distance, 0.0, 1.0)
        return np.exp(-1.0 * ratio * ratio).astype(np.float32)

    @staticmethod
    def _write_raster(path, array, gt, projection, gdal_type, nodata=None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(
            path,
            int(array.shape[1]),
            int(array.shape[0]),
            1,
            gdal_type,
            options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
        )
        if ds is None:
            raise QgsProcessingException(f"Δεν μπόρεσε να δημιουργηθεί το raster: {path}")
        ds.SetGeoTransform(gt)
        ds.SetProjection(projection)
        band = ds.GetRasterBand(1)
        if nodata is not None:
            band.SetNoDataValue(float(nodata))
        band.WriteArray(array)
        band.FlushCache()
        ds.FlushCache()
        ds = None

    @staticmethod
    def _angular_difference(bearings, azimuth):
        return np.abs((bearings.astype(np.float32) - float(azimuth) + 180.0) % 360.0 - 180.0)

    @staticmethod
    def _sector_geometry(x, y, azimuth, fov, radius):
        half = fov / 2.0
        if fov >= 359.999:
            angles = np.linspace(0.0, 360.0, 181)
            ring = []
            for angle in angles:
                rad = math.radians(float(angle))
                ring.append(QgsPointXY(x + radius * math.sin(rad), y + radius * math.cos(rad)))
            ring.append(ring[0])
            return QgsGeometry.fromPolygonXY([ring])

        start = azimuth - half
        end = azimuth + half
        segment_count = max(12, int(math.ceil(fov / 3.0)))
        angles = np.linspace(start, end, segment_count + 1)
        ring = [QgsPointXY(x, y)]
        for angle in angles:
            rad = math.radians(float(angle))
            ring.append(QgsPointXY(x + radius * math.sin(rad), y + radius * math.cos(rad)))
        ring.append(QgsPointXY(x, y))
        return QgsGeometry.fromPolygonXY([ring])

    def _build_candidates(
        self,
        candidate_source,
        dem_crs,
        transform_context,
        gt,
        rows,
        cols,
        dem,
        dem_valid,
        landcover,
        spacing,
        max_candidates,
        min_elevation,
        feedback,
    ):
        candidates = []
        seen_cells = set()

        def try_add(x, y):
            rc = self._xy_to_row_col(x, y, gt, rows, cols)
            if rc is None:
                return
            row, col = rc
            if (row, col) in seen_cells:
                return
            if not dem_valid[row, col]:
                return
            if float(dem[row, col]) < min_elevation:
                return
            # Reject permanent water. Other land classes remain possible tower sites.
            if int(round(float(landcover[row, col]))) == 80:
                return
            seen_cells.add((row, col))
            candidates.append({"x": float(x), "y": float(y), "row": row, "col": col})

        if candidate_source is not None:
            source_crs = candidate_source.sourceCrs()
            coord_transform = None
            if source_crs.isValid() and source_crs != dem_crs:
                coord_transform = QgsCoordinateTransform(source_crs, dem_crs, transform_context)

            for feature in candidate_source.getFeatures():
                if feedback.isCanceled():
                    break
                geometry = feature.geometry()
                if geometry is None or geometry.isEmpty():
                    continue
                point = geometry.asPoint() if not geometry.isMultipart() else geometry.asMultiPoint()[0]
                qpoint = QgsPointXY(point)
                if coord_transform is not None:
                    qpoint = coord_transform.transform(qpoint)
                try_add(qpoint.x(), qpoint.y())
        else:
            min_x = gt[0]
            max_x = gt[0] + cols * gt[1]
            max_y = gt[3]
            min_y = gt[3] + rows * gt[5]
            x_values = np.arange(min_x + spacing / 2.0, max_x, spacing)
            y_values = np.arange(min_y + spacing / 2.0, max_y, spacing)
            for y in y_values:
                for x in x_values:
                    try_add(float(x), float(y))

        if not candidates:
            raise QgsProcessingException(
                "Δεν βρέθηκαν έγκυρα υποψήφια σημεία. Έλεγξε το CRS, το DEM, το landcover και το ελάχιστο υψόμετρο."
            )

        if len(candidates) > max_candidates:
            # Deterministic, spatially distributed down-sampling of the regular list.
            keep_idx = np.linspace(0, len(candidates) - 1, max_candidates, dtype=int)
            candidates = [candidates[int(i)] for i in keep_idx]
            feedback.pushInfo(
                f"Τα υποψήφια σημεία περιορίστηκαν σε {max_candidates} για να παραμείνει πρακτικός ο χρόνος εκτέλεσης."
            )

        return candidates

    def _build_local_uphill_candidates(
        self,
        camera,
        gt,
        rows,
        cols,
        dem,
        dem_valid,
        landcover,
        radius,
        step,
        max_candidates,
    ):
        """Create nearby higher-terrain candidates around an initial camera.

        This is a robust version of "follow the uphill slope": instead of trusting
        one single gradient direction, it samples the local neighbourhood and keeps
        higher points. The expensive viewshed calculation is then run only for the
        highest local candidates.
        """
        if radius <= 0 or step <= 0:
            return []

        start_x = float(camera["x"])
        start_y = float(camera["y"])
        start_row = int(camera["row"])
        start_col = int(camera["col"])
        start_elev = float(dem[start_row, start_col])

        px = abs(float(gt[1]))
        py = abs(float(gt[5]))
        col_step = max(1, int(round(step / px)))
        row_step = max(1, int(round(step / py)))
        col_radius = int(math.ceil(radius / px))
        row_radius = int(math.ceil(radius / py))

        local = []
        seen = set()
        row0 = max(0, start_row - row_radius)
        row1 = min(rows - 1, start_row + row_radius)
        col0 = max(0, start_col - col_radius)
        col1 = min(cols - 1, start_col + col_radius)

        for row in range(row0, row1 + 1, row_step):
            for col in range(col0, col1 + 1, col_step):
                if (row, col) == (start_row, start_col) or (row, col) in seen:
                    continue
                seen.add((row, col))
                if not dem_valid[row, col]:
                    continue
                if int(round(float(landcover[row, col]))) == 80:
                    continue

                x = float(gt[0] + (col + 0.5) * gt[1])
                y = float(gt[3] + (row + 0.5) * gt[5])
                distance = math.hypot(x - start_x, y - start_y)
                if distance > radius + max(px, py):
                    continue

                elev = float(dem[row, col])
                if elev <= start_elev + 0.5:
                    continue

                local.append(
                    {
                        "x": x,
                        "y": y,
                        "row": row,
                        "col": col,
                        "elev_m": elev,
                        "distance_from_initial_m": distance,
                    }
                )

        # We intentionally prioritise high ground before running expensive viewsheds.
        # Distance is a secondary tie-breaker so equally high nearby points are tested first.
        local.sort(key=lambda c: (-c["elev_m"], c["distance_from_initial_m"]))
        if len(local) > max_candidates:
            local = local[:max_candidates]
        return local

    def _visibility_for_candidate(
        self,
        candidate,
        tag,
        dem_layer,
        dem_crs,
        observer_height,
        target_height,
        max_distance,
        risk_flat,
        gt,
        rows,
        cols,
        parameter_names,
        context,
        feedback,
        temp_dir,
    ):
        """Run one 360° GDAL viewshed and return target indices/bearings/quality."""
        output_path = os.path.join(temp_dir, f"refine_{tag}.tif")
        observer_string = f"{candidate['x']:.6f},{candidate['y']:.6f} [{dem_crs.authid()}]"
        view_params = {
            "INPUT": dem_layer,
            "BAND": 1,
            "OBSERVER": observer_string,
            "OBSERVER_HEIGHT": observer_height,
            "TARGET_HEIGHT": target_height,
            "MAX_DISTANCE": max_distance,
            "OUTPUT": output_path,
            "EXTRA": "",
        }
        if "CREATION_OPTIONS" in parameter_names:
            view_params["CREATION_OPTIONS"] = "COMPRESS=LZW|TILED=YES"
        elif "OPTIONS" in parameter_names:
            view_params["OPTIONS"] = "COMPRESS=LZW|TILED=YES"

        try:
            result = processing.run(
                "gdal:viewshed",
                view_params,
                context=context,
                feedback=feedback,
                is_child_algorithm=True,
            )
            result_path = result.get("OUTPUT", output_path)
            view_ds = gdal.Open(str(result_path), gdal.GA_ReadOnly)
            if view_ds is None:
                return None

            view_array = view_ds.GetRasterBand(1).ReadAsArray()
            visible_local = np.argwhere(view_array > 0)
            local_gt = view_ds.GetGeoTransform()
            view_ds = None
            if visible_local.size == 0:
                return None

            col_offset = int(round((local_gt[0] - gt[0]) / gt[1]))
            row_offset = int(round((local_gt[3] - gt[3]) / gt[5]))
            global_rows = visible_local[:, 0].astype(np.int64) + row_offset
            global_cols = visible_local[:, 1].astype(np.int64) + col_offset
            inside = (
                (global_rows >= 0)
                & (global_rows < rows)
                & (global_cols >= 0)
                & (global_cols < cols)
            )
            global_rows = global_rows[inside]
            global_cols = global_cols[inside]
            if global_rows.size == 0:
                return None

            flat_indices = (global_rows * cols + global_cols).astype(np.int32)
            target_mask = risk_flat[flat_indices] > 0
            flat_indices = flat_indices[target_mask]
            global_rows = global_rows[target_mask]
            global_cols = global_cols[target_mask]
            if flat_indices.size == 0:
                return None

            cell_x, cell_y = self._cell_centers(global_rows, global_cols, gt)
            dx = cell_x - candidate["x"]
            dy = cell_y - candidate["y"]
            distance = np.hypot(dx, dy)
            within = distance <= max_distance + max(abs(gt[1]), abs(gt[5]))
            flat_indices = flat_indices[within]
            dx = dx[within]
            dy = dy[within]
            distance = distance[within]
            if flat_indices.size == 0:
                return None

            bearings = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
            bearings = np.rint(bearings).astype(np.uint16)
            quality = self._distance_quality(distance, max_distance)
            return {
                "indices": flat_indices,
                "bearings": bearings,
                "quality": quality,
            }
        finally:
            for candidate_file in (output_path, output_path + ".aux.xml"):
                try:
                    if os.path.exists(candidate_file):
                        os.remove(candidate_file)
                except OSError:
                    pass

    def _best_orientation_against_baseline(
        self,
        item,
        baseline_quality,
        baseline_score,
        risk_flat,
        azimuths,
        fov,
        half_fov,
    ):
        """Find the azimuth giving the best whole-system weighted score."""
        best = None
        best_total_score = -1.0
        indices = item["indices"]
        bearings = item["bearings"]
        quality = item["quality"]
        existing = baseline_quality[indices]

        for azimuth in azimuths:
            if fov >= 359.999:
                mask = np.ones(indices.shape, dtype=bool)
            else:
                mask = self._angular_difference(bearings, azimuth) <= half_fov
            if not np.any(mask):
                continue

            selected_indices = indices[mask]
            selected_quality = quality[mask]
            improvement = selected_quality - existing[mask]
            positive = improvement > 1e-6
            gain = 0.0
            if np.any(positive):
                gain = float(
                    np.sum(
                        risk_flat[selected_indices[positive]] * improvement[positive],
                        dtype=np.float64,
                    )
                )
            total_score = baseline_score + gain
            if total_score > best_total_score:
                best_total_score = total_score
                best = {
                    "azimuth": float(azimuth),
                    "indices": selected_indices.copy(),
                    "quality": selected_quality.copy(),
                    "gain_against_baseline": gain,
                    "total_score": total_score,
                }
        return best

    @staticmethod
    def _compose_coverage(cameras, n_cells):
        quality = np.zeros(n_cells, dtype=np.float32)
        count = np.zeros(n_cells, dtype=np.uint16)
        for camera in cameras:
            indices = camera["indices"]
            values = camera["quality"]
            quality[indices] = np.maximum(quality[indices], values)
            count[indices] += 1
        return quality, count

    def processAlgorithm(self, parameters, context, feedback):
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
        landcover_layer = self.parameterAsRasterLayer(parameters, self.LANDCOVER, context)
        candidate_source = self.parameterAsSource(parameters, self.CANDIDATES, context)

        if dem_layer is None or landcover_layer is None:
            raise QgsProcessingException("Πρέπει να επιλεγούν DEM και landcover raster.")

        num_cameras = self.parameterAsInt(parameters, self.NUM_CAMERAS, context)
        fov = self.parameterAsDouble(parameters, self.FOV, context)
        az_step = self.parameterAsDouble(parameters, self.AZ_STEP, context)
        observer_height = self.parameterAsDouble(parameters, self.OBSERVER_HEIGHT, context)
        target_height = self.parameterAsDouble(parameters, self.TARGET_HEIGHT, context)
        max_distance = self.parameterAsDouble(parameters, self.MAX_DISTANCE, context)
        grid_spacing = self.parameterAsDouble(parameters, self.GRID_SPACING, context)
        max_candidates = self.parameterAsInt(parameters, self.MAX_CANDIDATES, context)
        min_elevation = self.parameterAsDouble(parameters, self.MIN_ELEVATION, context)
        refine_radius = self.parameterAsDouble(parameters, self.REFINE_RADIUS, context)
        refine_step = self.parameterAsDouble(parameters, self.REFINE_STEP, context)
        refine_max_loss = self.parameterAsDouble(parameters, self.REFINE_MAX_LOSS, context)
        refine_max_candidates = self.parameterAsInt(parameters, self.REFINE_MAX_CANDIDATES, context)

        dem_path = dem_layer.source().split("|")[0]
        landcover_path = landcover_layer.source().split("|")[0]
        dem_ds = gdal.Open(dem_path, gdal.GA_ReadOnly)
        lc_ds = gdal.Open(landcover_path, gdal.GA_ReadOnly)
        if dem_ds is None or lc_ds is None:
            raise QgsProcessingException("Αποτυχία ανοίγματος ενός από τα raster με GDAL.")

        if not self._same_grid(dem_ds, lc_ds):
            raise QgsProcessingException(
                "DEM και landcover δεν έχουν ακριβώς ίδιο grid. Τρέξε ξανά Align rasters με το DEM ως reference."
            )

        dem_crs = dem_layer.crs()
        if dem_crs.isGeographic():
            raise QgsProcessingException(
                "Το DEM είναι σε γεωγραφικό CRS. Χρειάζεται projected CRS σε μέτρα, π.χ. EPSG:32635."
            )

        rows = dem_ds.RasterYSize
        cols = dem_ds.RasterXSize
        gt = dem_ds.GetGeoTransform()
        projection = dem_ds.GetProjection()
        if gt[1] <= 0 or gt[5] >= 0:
            raise QgsProcessingException("Μη αναμενόμενο raster grid/geotransform.")

        dem_band = dem_ds.GetRasterBand(1)
        lc_band = lc_ds.GetRasterBand(1)
        dem = dem_band.ReadAsArray().astype(np.float32)
        landcover = lc_band.ReadAsArray()

        dem_nodata = dem_band.GetNoDataValue()
        dem_valid = np.isfinite(dem)
        if dem_nodata is not None:
            dem_valid &= ~np.isclose(dem, float(dem_nodata))

        lc_valid = np.isfinite(landcover.astype(np.float32))
        lc_nodata = lc_band.GetNoDataValue()
        if lc_nodata is not None:
            lc_valid &= ~np.isclose(landcover.astype(np.float32), float(lc_nodata))

        risk = np.zeros((rows, cols), dtype=np.float32)
        lc_int = np.rint(landcover).astype(np.int16, copy=False)
        for code, weight in self.WORLDCOVER_WEIGHTS.items():
            risk[lc_int == code] = float(weight)

        risk[~lc_valid] = 0.0
        risk[~dem_valid] = 0.0
        risk_flat = risk.ravel()
        total_risk = float(risk_flat.sum(dtype=np.float64))
        if total_risk <= 0:
            raise QgsProcessingException(
                "Ο χάρτης βλάστησης δεν περιέχει αναγνωρίσιμες WorldCover τιμές ή δεν είναι ευθυγραμμισμένος."
            )

        feedback.pushInfo("Δημιουργία υποψήφιων θέσεων…")
        candidates = self._build_candidates(
            candidate_source,
            dem_crs,
            context.transformContext(),
            gt,
            rows,
            cols,
            dem,
            dem_valid,
            landcover,
            grid_spacing,
            max_candidates,
            min_elevation,
            feedback,
        )
        feedback.pushInfo(f"Έγκυρες υποψήφιες θέσεις: {len(candidates)}")

        alg = QgsApplication.processingRegistry().algorithmById("gdal:viewshed")
        if alg is None:
            raise QgsProcessingException(
                "Δεν βρέθηκε το gdal:viewshed. Ενεργοποίησε τον GDAL provider στο Processing."
            )
        parameter_names = {definition.name() for definition in alg.parameterDefinitions()}

        temp_dir = tempfile.mkdtemp(prefix="firecam_viewshed_")
        cache = []
        usable_candidates = []

        try:
            feedback.pushInfo("Υπολογισμός ενός viewshed 360° για κάθε υποψήφια θέση…")
            for index, candidate in enumerate(candidates):
                if feedback.isCanceled():
                    break

                feedback.setProgress(5.0 + 60.0 * index / max(1, len(candidates)))
                output_path = os.path.join(temp_dir, f"viewshed_{index:05d}.tif")
                observer_string = (
                    f"{candidate['x']:.6f},{candidate['y']:.6f} [{dem_crs.authid()}]"
                )
                view_params = {
                    "INPUT": dem_layer,
                    "BAND": 1,
                    "OBSERVER": observer_string,
                    "OBSERVER_HEIGHT": observer_height,
                    "TARGET_HEIGHT": target_height,
                    "MAX_DISTANCE": max_distance,
                    "OUTPUT": output_path,
                    "EXTRA": "",
                }
                if "CREATION_OPTIONS" in parameter_names:
                    view_params["CREATION_OPTIONS"] = "COMPRESS=LZW|TILED=YES"
                elif "OPTIONS" in parameter_names:
                    view_params["OPTIONS"] = "COMPRESS=LZW|TILED=YES"

                try:
                    result = processing.run(
                        "gdal:viewshed",
                        view_params,
                        context=context,
                        feedback=feedback,
                        is_child_algorithm=True,
                    )
                    result_path = result.get("OUTPUT", output_path)
                    view_ds = gdal.Open(str(result_path), gdal.GA_ReadOnly)
                    if view_ds is None:
                        feedback.reportError(f"Αποτυχία viewshed για υποψήφιο {index + 1}.")
                        continue

                    view_array = view_ds.GetRasterBand(1).ReadAsArray()
                    visible_local = np.argwhere(view_array > 0)
                    local_gt = view_ds.GetGeoTransform()
                    view_ds = None

                    if visible_local.size == 0:
                        continue

                    col_offset = int(round((local_gt[0] - gt[0]) / gt[1]))
                    row_offset = int(round((local_gt[3] - gt[3]) / gt[5]))
                    global_rows = visible_local[:, 0].astype(np.int64) + row_offset
                    global_cols = visible_local[:, 1].astype(np.int64) + col_offset
                    inside = (
                        (global_rows >= 0)
                        & (global_rows < rows)
                        & (global_cols >= 0)
                        & (global_cols < cols)
                    )
                    global_rows = global_rows[inside]
                    global_cols = global_cols[inside]
                    if global_rows.size == 0:
                        continue

                    flat_indices = (global_rows * cols + global_cols).astype(np.int32)
                    target_mask = risk_flat[flat_indices] > 0
                    flat_indices = flat_indices[target_mask]
                    global_rows = global_rows[target_mask]
                    global_cols = global_cols[target_mask]
                    if flat_indices.size == 0:
                        continue

                    cell_x, cell_y = self._cell_centers(global_rows, global_cols, gt)
                    dx = cell_x - candidate["x"]
                    dy = cell_y - candidate["y"]
                    distance = np.hypot(dx, dy)
                    within = distance <= max_distance + max(abs(gt[1]), abs(gt[5]))
                    flat_indices = flat_indices[within]
                    dx = dx[within]
                    dy = dy[within]
                    distance = distance[within]
                    if flat_indices.size == 0:
                        continue

                    bearings = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
                    bearings = np.rint(bearings).astype(np.uint16)
                    quality = self._distance_quality(distance, max_distance)

                    cache.append(
                        {
                            "indices": flat_indices,
                            "bearings": bearings,
                            "quality": quality,
                        }
                    )
                    usable_candidates.append(candidate)

                except Exception as exc:
                    feedback.reportError(
                        f"Το viewshed απέτυχε για τον υποψήφιο {index + 1}: {exc}"
                    )
                finally:
                    for candidate_file in (output_path, output_path + ".aux.xml"):
                        try:
                            if os.path.exists(candidate_file):
                                os.remove(candidate_file)
                        except OSError:
                            pass

            if feedback.isCanceled():
                return {}

            if not cache:
                raise QgsProcessingException(
                    "Κανένα υποψήφιο σημείο δεν παρήγαγε ορατή περιοχή-στόχο."
                )

            feedback.pushInfo(f"Υποψήφια σημεία με χρήσιμη ορατότητα: {len(cache)}")

            azimuths = np.arange(0.0, 360.0, az_step, dtype=np.float32)
            if azimuths.size == 0:
                azimuths = np.array([0.0], dtype=np.float32)
            half_fov = fov / 2.0

            current_quality = np.zeros(rows * cols, dtype=np.float32)
            coverage_count = np.zeros(rows * cols, dtype=np.uint16)
            remaining = set(range(len(cache)))
            selected = []

            feedback.pushInfo("Greedy επιλογή θέσεων και κατευθύνσεων…")
            for rank in range(1, num_cameras + 1):
                if feedback.isCanceled():
                    break
                best = None
                best_gain = 0.0

                for position, candidate_index in enumerate(list(remaining)):
                    item = cache[candidate_index]
                    indices = item["indices"]
                    bearings = item["bearings"]
                    quality = item["quality"]
                    existing = current_quality[indices]

                    for azimuth in azimuths:
                        if fov >= 359.999:
                            mask = np.ones(indices.shape, dtype=bool)
                        else:
                            mask = self._angular_difference(bearings, azimuth) <= half_fov
                        if not np.any(mask):
                            continue
                        selected_indices = indices[mask]
                        improvement = quality[mask] - existing[mask]
                        positive = improvement > 1e-6
                        if not np.any(positive):
                            continue
                        gain = float(
                            np.sum(
                                risk_flat[selected_indices[positive]] * improvement[positive],
                                dtype=np.float64,
                            )
                        )
                        if gain > best_gain:
                            best_gain = gain
                            best = (candidate_index, float(azimuth), mask)

                    if position % 20 == 0:
                        progress = 66.0 + 24.0 * (
                            ((rank - 1) + position / max(1, len(remaining)))
                            / max(1, num_cameras)
                        )
                        feedback.setProgress(min(progress, 90.0))

                if best is None or best_gain <= 0:
                    feedback.pushWarning(
                        "Δεν υπάρχει επιπλέον συνδυασμός που να προσθέτει νέα κάλυψη."
                    )
                    break

                candidate_index, azimuth, mask = best
                item = cache[candidate_index]
                indices = item["indices"][mask]
                quality = item["quality"][mask]
                improvement_mask = quality > current_quality[indices]
                current_quality[indices[improvement_mask]] = quality[improvement_mask]
                coverage_count[indices] += 1
                remaining.remove(candidate_index)

                effective_pct = 100.0 * float(
                    np.sum(risk_flat * current_quality, dtype=np.float64) / total_risk
                )
                binary_pct = 100.0 * float(
                    np.sum(risk_flat[coverage_count > 0], dtype=np.float64) / total_risk
                )
                selected.append(
                    {
                        "rank": rank,
                        "candidate_index": candidate_index,
                        "x": usable_candidates[candidate_index]["x"],
                        "y": usable_candidates[candidate_index]["y"],
                        "row": usable_candidates[candidate_index]["row"],
                        "col": usable_candidates[candidate_index]["col"],
                        "elev_m": float(
                            dem[
                                usable_candidates[candidate_index]["row"],
                                usable_candidates[candidate_index]["col"],
                            ]
                        ),
                        "azimuth": azimuth,
                        "gain": best_gain,
                        "effective_pct": effective_pct,
                        "binary_pct": binary_pct,
                        "indices": indices.copy(),
                        "quality": quality.copy(),
                    }
                )
                feedback.pushInfo(
                    f"{rank}. θέση ({selected[-1]['x']:.1f}, {selected[-1]['y']:.1f}), "
                    f"azimuth {azimuth:.0f}°, σταθμισμένη κάλυψη {effective_pct:.2f}%"
                )

            if not selected:
                raise QgsProcessingException("Δεν επιλέχθηκε καμία κάμερα.")

            # Keep a scalar snapshot so the user can see exactly where the global
            # optimiser originally placed each camera before local refinement.
            initial_selected = []
            for camera in selected:
                initial_selected.append(
                    {
                        "rank": camera["rank"],
                        "x": camera["x"],
                        "y": camera["y"],
                        "row": camera["row"],
                        "col": camera["col"],
                        "elev_m": camera["elev_m"],
                        "azimuth": camera["azimuth"],
                        "gain": camera["gain"],
                        "effective_pct": camera["effective_pct"],
                        "binary_pct": camera["binary_pct"],
                    }
                )

            refined_selected = []
            for camera in selected:
                copied = dict(camera)
                copied["indices"] = camera["indices"].copy()
                copied["quality"] = camera["quality"].copy()
                copied["init_x"] = camera["x"]
                copied["init_y"] = camera["y"]
                copied["init_elev_m"] = camera["elev_m"]
                copied["init_azimuth"] = camera["azimuth"]
                copied["move_m"] = 0.0
                copied["elev_gain_m"] = 0.0
                copied["refined"] = 0
                refined_selected.append(copied)

            initial_system_quality, _ = self._compose_coverage(
                refined_selected, rows * cols
            )
            initial_system_score = float(
                np.sum(risk_flat * initial_system_quality, dtype=np.float64)
            )
            global_score_floor = initial_system_score * (1.0 - refine_max_loss / 100.0)

            if refine_radius > 0.0:
                feedback.pushInfo(
                    "Τοπικό uphill refinement: αναζήτηση υψηλότερων σημείων γύρω από κάθε αρχική θέση…"
                )

                for camera_pos in range(len(refined_selected)):
                    if feedback.isCanceled():
                        break

                    current_camera = refined_selected[camera_pos]
                    current_all_quality, _ = self._compose_coverage(
                        refined_selected, rows * cols
                    )
                    current_system_score = float(
                        np.sum(risk_flat * current_all_quality, dtype=np.float64)
                    )

                    # Coverage supplied by all other cameras. The camera being refined
                    # is then tested against this fixed baseline.
                    others = [
                        cam for idx, cam in enumerate(refined_selected) if idx != camera_pos
                    ]
                    baseline_quality, _ = self._compose_coverage(others, rows * cols)
                    baseline_score = float(
                        np.sum(risk_flat * baseline_quality, dtype=np.float64)
                    )

                    local_candidates = self._build_local_uphill_candidates(
                        current_camera,
                        gt,
                        rows,
                        cols,
                        dem,
                        dem_valid,
                        landcover,
                        refine_radius,
                        refine_step,
                        refine_max_candidates,
                    )

                    # Never allow cumulative refinement to push the whole system below
                    # the original global score floor, and also limit each individual move.
                    required_score = max(
                        global_score_floor,
                        current_system_score * (1.0 - refine_max_loss / 100.0),
                    )

                    best_replacement = None
                    best_elevation = float(current_camera["elev_m"])
                    best_total_score = current_system_score

                    for local_index, candidate in enumerate(local_candidates):
                        if feedback.isCanceled():
                            break

                        item = self._visibility_for_candidate(
                            candidate,
                            f"r{camera_pos + 1:02d}_{local_index:03d}",
                            dem_layer,
                            dem_crs,
                            observer_height,
                            target_height,
                            max_distance,
                            risk_flat,
                            gt,
                            rows,
                            cols,
                            parameter_names,
                            context,
                            feedback,
                            temp_dir,
                        )
                        if item is None:
                            continue

                        orientation = self._best_orientation_against_baseline(
                            item,
                            baseline_quality,
                            baseline_score,
                            risk_flat,
                            azimuths,
                            fov,
                            half_fov,
                        )
                        if orientation is None:
                            continue

                        total_score = float(orientation["total_score"])
                        elevation = float(candidate["elev_m"])
                        if total_score + 1e-9 < required_score:
                            continue

                        # Uphill preference: among positions that preserve the required
                        # system score, choose the highest terrain. Score breaks ties.
                        if (
                            elevation > best_elevation + 0.5
                            or (
                                abs(elevation - best_elevation) <= 0.5
                                and total_score > best_total_score
                            )
                        ):
                            best_elevation = elevation
                            best_total_score = total_score
                            best_replacement = {
                                **candidate,
                                **orientation,
                            }

                    if best_replacement is not None:
                        old_x = float(current_camera["x"])
                        old_y = float(current_camera["y"])
                        old_elev = float(current_camera["elev_m"])

                        current_camera["x"] = float(best_replacement["x"])
                        current_camera["y"] = float(best_replacement["y"])
                        current_camera["row"] = int(best_replacement["row"])
                        current_camera["col"] = int(best_replacement["col"])
                        current_camera["elev_m"] = float(best_replacement["elev_m"])
                        current_camera["azimuth"] = float(best_replacement["azimuth"])
                        current_camera["indices"] = best_replacement["indices"].copy()
                        current_camera["quality"] = best_replacement["quality"].copy()
                        current_camera["move_m"] = math.hypot(
                            current_camera["x"] - old_x,
                            current_camera["y"] - old_y,
                        )
                        current_camera["elev_gain_m"] = current_camera["elev_m"] - old_elev
                        current_camera["refined"] = 1

                        feedback.pushInfo(
                            f"Κάμερα {current_camera['rank']}: {old_elev:.1f} → "
                            f"{current_camera['elev_m']:.1f} m, μετακίνηση "
                            f"{current_camera['move_m']:.0f} m, system score "
                            f"{100.0 * best_total_score / max(initial_system_score, 1e-12):.2f}% του αρχικού."
                        )
                    else:
                        feedback.pushInfo(
                            f"Κάμερα {current_camera['rank']}: παραμένει στην αρχική θέση "
                            f"({current_camera['elev_m']:.1f} m)."
                        )

            # Recompute final system coverage from the refined camera set.
            current_quality, coverage_count = self._compose_coverage(
                refined_selected, rows * cols
            )

            # Recompute marginal metrics in rank order for the final positions.
            running_quality = np.zeros(rows * cols, dtype=np.float32)
            running_count = np.zeros(rows * cols, dtype=np.uint16)
            for camera in refined_selected:
                indices = camera["indices"]
                values = camera["quality"]
                existing = running_quality[indices]
                improvement = values - existing
                positive = improvement > 1e-6
                camera["gain"] = 0.0
                if np.any(positive):
                    camera["gain"] = float(
                        np.sum(
                            risk_flat[indices[positive]] * improvement[positive],
                            dtype=np.float64,
                        )
                    )
                running_quality[indices] = np.maximum(existing, values)
                running_count[indices] += 1
                camera["effective_pct"] = 100.0 * float(
                    np.sum(risk_flat * running_quality, dtype=np.float64) / total_risk
                )
                camera["binary_pct"] = 100.0 * float(
                    np.sum(risk_flat[running_count > 0], dtype=np.float64) / total_risk
                )

            # ---------- Vector outputs: initial, final and movement lines ----------
            initial_fields = QgsFields()
            initial_fields.append(QgsField("rank", QVariant.Int))
            initial_fields.append(QgsField("azimuth", QVariant.Double))
            initial_fields.append(QgsField("elev_m", QVariant.Double))
            initial_fields.append(QgsField("eff_pct", QVariant.Double))
            initial_fields.append(QgsField("bin_pct", QVariant.Double))

            final_fields = QgsFields()
            final_fields.append(QgsField("rank", QVariant.Int))
            final_fields.append(QgsField("azimuth", QVariant.Double))
            final_fields.append(QgsField("fov_deg", QVariant.Double))
            final_fields.append(QgsField("gain", QVariant.Double))
            final_fields.append(QgsField("eff_pct", QVariant.Double))
            final_fields.append(QgsField("bin_pct", QVariant.Double))
            final_fields.append(QgsField("obs_h_m", QVariant.Double))
            final_fields.append(QgsField("target_h_m", QVariant.Double))
            final_fields.append(QgsField("range_m", QVariant.Double))
            final_fields.append(QgsField("elev_m", QVariant.Double))
            final_fields.append(QgsField("init_elev", QVariant.Double))
            final_fields.append(QgsField("move_m", QVariant.Double))
            final_fields.append(QgsField("elev_gain", QVariant.Double))
            final_fields.append(QgsField("refined", QVariant.Int))

            move_fields = QgsFields()
            move_fields.append(QgsField("rank", QVariant.Int))
            move_fields.append(QgsField("move_m", QVariant.Double))
            move_fields.append(QgsField("elev_gain", QVariant.Double))
            move_fields.append(QgsField("init_elev", QVariant.Double))
            move_fields.append(QgsField("final_elev", QVariant.Double))
            move_fields.append(QgsField("init_az", QVariant.Double))
            move_fields.append(QgsField("final_az", QVariant.Double))

            initial_sink, initial_dest = self.parameterAsSink(
                parameters,
                self.OUT_INITIAL_CAMERAS,
                context,
                initial_fields,
                QgsWkbTypes.Point,
                dem_crs,
            )
            camera_sink, camera_dest = self.parameterAsSink(
                parameters,
                self.OUT_CAMERAS,
                context,
                final_fields,
                QgsWkbTypes.Point,
                dem_crs,
            )
            move_sink, move_dest = self.parameterAsSink(
                parameters,
                self.OUT_MOVES,
                context,
                move_fields,
                QgsWkbTypes.LineString,
                dem_crs,
            )
            fov_sink, fov_dest = self.parameterAsSink(
                parameters,
                self.OUT_FOV,
                context,
                final_fields,
                QgsWkbTypes.Polygon,
                dem_crs,
            )
            if initial_sink is None or camera_sink is None or move_sink is None or fov_sink is None:
                raise QgsProcessingException("Αποτυχία δημιουργίας vector εξόδων.")

            for initial in initial_selected:
                feature = QgsFeature(initial_fields)
                feature.setGeometry(
                    QgsGeometry.fromPointXY(QgsPointXY(initial["x"], initial["y"]))
                )
                feature.setAttributes(
                    [
                        initial["rank"],
                        initial["azimuth"],
                        initial["elev_m"],
                        initial["effective_pct"],
                        initial["binary_pct"],
                    ]
                )
                initial_sink.addFeature(feature, QgsFeatureSink.FastInsert)

            for camera in refined_selected:
                attributes = [
                    camera["rank"],
                    camera["azimuth"],
                    fov,
                    camera["gain"],
                    camera["effective_pct"],
                    camera["binary_pct"],
                    observer_height,
                    target_height,
                    max_distance,
                    camera["elev_m"],
                    camera["init_elev_m"],
                    camera["move_m"],
                    camera["elev_gain_m"],
                    camera["refined"],
                ]

                point_feature = QgsFeature(final_fields)
                point_feature.setGeometry(
                    QgsGeometry.fromPointXY(QgsPointXY(camera["x"], camera["y"]))
                )
                point_feature.setAttributes(attributes)
                camera_sink.addFeature(point_feature, QgsFeatureSink.FastInsert)

                fov_feature = QgsFeature(final_fields)
                fov_feature.setGeometry(
                    self._sector_geometry(
                        camera["x"], camera["y"], camera["azimuth"], fov, max_distance
                    )
                )
                fov_feature.setAttributes(attributes)
                fov_sink.addFeature(fov_feature, QgsFeatureSink.FastInsert)

                move_feature = QgsFeature(move_fields)
                move_feature.setGeometry(
                    QgsGeometry.fromPolylineXY(
                        [
                            QgsPointXY(camera["init_x"], camera["init_y"]),
                            QgsPointXY(camera["x"], camera["y"]),
                        ]
                    )
                )
                move_feature.setAttributes(
                    [
                        camera["rank"],
                        camera["move_m"],
                        camera["elev_gain_m"],
                        camera["init_elev_m"],
                        camera["elev_m"],
                        camera["init_azimuth"],
                        camera["azimuth"],
                    ]
                )
                move_sink.addFeature(move_feature, QgsFeatureSink.FastInsert)

            # ---------- Raster outputs based on FINAL positions ----------
            output_quality = self.parameterAsOutputLayer(parameters, self.OUT_QUALITY, context)
            output_count = self.parameterAsOutputLayer(parameters, self.OUT_COUNT, context)
            output_remaining = self.parameterAsOutputLayer(
                parameters, self.OUT_REMAINING_RISK, context
            )
            summary_path = self.parameterAsFileOutput(parameters, self.OUT_SUMMARY, context)

            quality_2d = current_quality.reshape((rows, cols)).copy()
            count_2d = coverage_count.reshape((rows, cols)).copy()
            remaining_risk = risk * (1.0 - quality_2d)
            quality_2d[~dem_valid] = -9999.0
            count_2d[~dem_valid] = 0
            remaining_risk[~dem_valid] = -9999.0

            self._write_raster(
                output_quality,
                quality_2d.astype(np.float32),
                gt,
                projection,
                gdal.GDT_Float32,
                nodata=-9999,
            )
            self._write_raster(
                output_count,
                count_2d.astype(np.uint16),
                gt,
                projection,
                gdal.GDT_UInt16,
                nodata=0,
            )
            self._write_raster(
                output_remaining,
                remaining_risk.astype(np.float32),
                gt,
                projection,
                gdal.GDT_Float32,
                nodata=-9999,
            )

            final_effective = 100.0 * float(
                np.sum(risk_flat * current_quality, dtype=np.float64) / total_risk
            )
            final_binary = 100.0 * float(
                np.sum(risk_flat[coverage_count > 0], dtype=np.float64) / total_risk
            )
            final_system_score = float(
                np.sum(risk_flat * current_quality, dtype=np.float64)
            )

            os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
            with open(summary_path, "w", newline="", encoding="utf-8-sig") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["setting", "value"])
                writer.writerow(["candidate_count", len(cache)])
                writer.writerow(["selected_camera_count", len(refined_selected)])
                writer.writerow(["horizontal_fov_deg", fov])
                writer.writerow(["azimuth_step_deg", az_step])
                writer.writerow(["observer_height_m", observer_height])
                writer.writerow(["target_height_m", target_height])
                writer.writerow(["max_distance_m", max_distance])
                writer.writerow(["refine_radius_m", refine_radius])
                writer.writerow(["refine_step_m", refine_step])
                writer.writerow(["refine_max_loss_pct", refine_max_loss])
                writer.writerow(["refine_max_candidates_per_camera", refine_max_candidates])
                writer.writerow(["initial_system_score", initial_system_score])
                writer.writerow(["final_system_score", final_system_score])
                writer.writerow([
                    "final_score_vs_initial_pct",
                    100.0 * final_system_score / max(initial_system_score, 1e-12),
                ])
                writer.writerow(["effective_weighted_coverage_pct", final_effective])
                writer.writerow(["binary_weighted_coverage_pct", final_binary])
                writer.writerow([])
                writer.writerow(
                    [
                        "rank",
                        "initial_x",
                        "initial_y",
                        "initial_elev_m",
                        "final_x",
                        "final_y",
                        "final_elev_m",
                        "move_m",
                        "elevation_gain_m",
                        "initial_azimuth_deg",
                        "final_azimuth_deg",
                        "refined",
                        "marginal_gain",
                        "effective_pct",
                        "binary_pct",
                    ]
                )
                for camera in refined_selected:
                    writer.writerow(
                        [
                            camera["rank"],
                            camera["init_x"],
                            camera["init_y"],
                            camera["init_elev_m"],
                            camera["x"],
                            camera["y"],
                            camera["elev_m"],
                            camera["move_m"],
                            camera["elev_gain_m"],
                            camera["init_azimuth"],
                            camera["azimuth"],
                            camera["refined"],
                            camera["gain"],
                            camera["effective_pct"],
                            camera["binary_pct"],
                        ]
                    )

            feedback.setProgress(100.0)
            feedback.pushInfo(
                f"Ολοκληρώθηκε: {len(refined_selected)} κάμερες, "
                f"τελική σταθμισμένη αποτελεσματική κάλυψη {final_effective:.2f}%, "
                f"τελικό score {100.0 * final_system_score / max(initial_system_score, 1e-12):.2f}% "
                f"του αρχικού coarse score."
            )

            return {
                self.OUT_INITIAL_CAMERAS: initial_dest,
                self.OUT_CAMERAS: camera_dest,
                self.OUT_MOVES: move_dest,
                self.OUT_FOV: fov_dest,
                self.OUT_QUALITY: output_quality,
                self.OUT_COUNT: output_count,
                self.OUT_REMAINING_RISK: output_remaining,
                self.OUT_SUMMARY: summary_path,
            }

        finally:
            dem_ds = None
            lc_ds = None
            shutil.rmtree(temp_dir, ignore_errors=True)
