# -*- coding: utf-8 -*-
"""
QGIS Processing Script: Lesvos Fire Camera Optimizer

Finds a greedy near-optimal set of camera locations and azimuths using:
- a projected DEM (metres),
- an aligned ESA WorldCover raster,
- GDAL viewshed,
- a horizontal field of view (default 180 degrees),
- land-cover risk weights,
- distance-dependent detection quality.

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

    OUT_CAMERAS = "OUT_CAMERAS"
    OUT_FOV = "OUT_FOV"
    OUT_QUALITY = "OUT_QUALITY"
    OUT_COUNT = "OUT_COUNT"
    OUT_REMAINING_RISK = "OUT_REMAINING_RISK"
    OUT_SUMMARY = "OUT_SUMMARY"

    # ESA WorldCover 2021 codes -> relative fire-monitoring importance.
    # These are modelling assumptions, not official fire-risk coefficients.
    WORLDCOVER_WEIGHTS = {
        10: 20.0,   # Δενδρώδης κάλυψη / δασικές εκτάσεις
        20: 10.0,   # Θαμνώδεις εκτάσεις
        30: 8.0,    # Χορτολιβαδικές εκτάσεις
        40: 5.0,    # Καλλιεργούμενες εκτάσεις
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
        return "lesvos_fire_camera_optimizer"

    def displayName(self):
        return self.tr("Βελτιστοποίηση θέσεων καμερών πυρανίχνευσης")

    def group(self):
        return self.tr("Πυρανίχνευση")

    def groupId(self):
        return "fire_detection"

    def shortHelpString(self):
        return self.tr(
            "Επιλέγει θέσεις και κατευθύνσεις καμερών με βάση DEM, WorldCover, "
            "viewshed, οριζόντιο FOV και απόσταση. Για πρώτη δοκιμή στη Λέσβο: "
            "FOV=180°, ύψος κάμερας=10 m, ύψος στόχου=20 m, εμβέλεια=4000 m, "
            "βήμα υποψήφιων θέσεων=3000 m, βήμα azimuth=30°. Τα δύο raster "
            "πρέπει να είναι ευθυγραμμισμένα, στο ίδιο projected CRS και σε μέτρα."
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
                defaultValue=30.0,
                minValue=1.0,
                maxValue=180.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OBSERVER_HEIGHT,
                self.tr("Ύψος κάμερας πάνω από το έδαφος (m)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
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
                defaultValue=3000.0,
                minValue=50.0,
                maxValue=20000.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_CANDIDATES,
                self.tr("Μέγιστος αριθμός υποψήφιων θέσεων"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=200,
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
            QgsProcessingParameterFeatureSink(
                self.OUT_CAMERAS,
                self.tr("Επιλεγμένες κάμερες"),
                QgsProcessing.TypeVectorPoint,
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
        return np.exp(-2.0 * ratio * ratio).astype(np.float32)

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
                        "azimuth": azimuth,
                        "gain": best_gain,
                        "effective_pct": effective_pct,
                        "binary_pct": binary_pct,
                    }
                )
                feedback.pushInfo(
                    f"{rank}. θέση ({selected[-1]['x']:.1f}, {selected[-1]['y']:.1f}), "
                    f"azimuth {azimuth:.0f}°, σταθμισμένη κάλυψη {effective_pct:.2f}%"
                )

            if not selected:
                raise QgsProcessingException("Δεν επιλέχθηκε καμία κάμερα.")

            # Prepare vector outputs.
            fields = QgsFields()
            fields.append(QgsField("rank", QVariant.Int))
            fields.append(QgsField("azimuth", QVariant.Double))
            fields.append(QgsField("fov_deg", QVariant.Double))
            fields.append(QgsField("gain", QVariant.Double))
            fields.append(QgsField("eff_pct", QVariant.Double))
            fields.append(QgsField("bin_pct", QVariant.Double))
            fields.append(QgsField("obs_h_m", QVariant.Double))
            fields.append(QgsField("target_h_m", QVariant.Double))
            fields.append(QgsField("range_m", QVariant.Double))

            camera_sink, camera_dest = self.parameterAsSink(
                parameters,
                self.OUT_CAMERAS,
                context,
                fields,
                QgsWkbTypes.Point,
                dem_crs,
            )
            fov_sink, fov_dest = self.parameterAsSink(
                parameters,
                self.OUT_FOV,
                context,
                fields,
                QgsWkbTypes.Polygon,
                dem_crs,
            )
            if camera_sink is None or fov_sink is None:
                raise QgsProcessingException("Αποτυχία δημιουργίας vector εξόδων.")

            for camera in selected:
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
                ]

                point_feature = QgsFeature(fields)
                point_feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(camera["x"], camera["y"])))
                point_feature.setAttributes(attributes)
                camera_sink.addFeature(point_feature, QgsFeatureSink.FastInsert)

                fov_feature = QgsFeature(fields)
                fov_feature.setGeometry(
                    self._sector_geometry(
                        camera["x"], camera["y"], camera["azimuth"], fov, max_distance
                    )
                )
                fov_feature.setAttributes(attributes)
                fov_sink.addFeature(fov_feature, QgsFeatureSink.FastInsert)

            # Raster outputs.
            output_quality = self.parameterAsOutputLayer(parameters, self.OUT_QUALITY, context)
            output_count = self.parameterAsOutputLayer(parameters, self.OUT_COUNT, context)
            output_remaining = self.parameterAsOutputLayer(
                parameters, self.OUT_REMAINING_RISK, context
            )
            summary_path = self.parameterAsFileOutput(parameters, self.OUT_SUMMARY, context)

            quality_2d = current_quality.reshape((rows, cols))
            count_2d = coverage_count.reshape((rows, cols))
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

            os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
            final_effective = selected[-1]["effective_pct"]
            final_binary = selected[-1]["binary_pct"]
            with open(summary_path, "w", newline="", encoding="utf-8-sig") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["setting", "value"])
                writer.writerow(["candidate_count", len(cache)])
                writer.writerow(["selected_camera_count", len(selected)])
                writer.writerow(["horizontal_fov_deg", fov])
                writer.writerow(["azimuth_step_deg", az_step])
                writer.writerow(["observer_height_m", observer_height])
                writer.writerow(["target_height_m", target_height])
                writer.writerow(["max_distance_m", max_distance])
                writer.writerow(["effective_weighted_coverage_pct", final_effective])
                writer.writerow(["binary_weighted_coverage_pct", final_binary])
                writer.writerow([])
                writer.writerow(
                    ["rank", "x", "y", "azimuth_deg", "marginal_gain", "effective_pct", "binary_pct"]
                )
                for camera in selected:
                    writer.writerow(
                        [
                            camera["rank"],
                            camera["x"],
                            camera["y"],
                            camera["azimuth"],
                            camera["gain"],
                            camera["effective_pct"],
                            camera["binary_pct"],
                        ]
                    )

            feedback.setProgress(100.0)
            feedback.pushInfo(
                f"Ολοκληρώθηκε: {len(selected)} κάμερες, "
                f"σταθμισμένη αποτελεσματική κάλυψη {final_effective:.2f}%, "
                f"ορατή σταθμισμένη περιοχή {final_binary:.2f}%."
            )

            return {
                self.OUT_CAMERAS: camera_dest,
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
