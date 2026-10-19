# -*- coding: utf-8 -*-
"""
QGIS Processing Script: Lesvos Shoulder Candidate Generator v1

Standalone terrain-only diagnostic tool.
It does NOT run viewsheds and does NOT optimize camera locations.
It only detects terrain shoulder/ridge-like candidate points from a DEM,
using the same basic terrain logic as the hybrid camera optimizer v7.

For each sampled terrain point it checks 8 directions (every 45 degrees) and
looks for a sufficiently steep downhill direction, a milder opposite/back
slope, a sufficient terrain break, and an acceptable local site slope.
Candidates are ranked by a terrain score and then filtered by a minimum
spacing.

Optional inputs:
- target polygon + buffer, to restrict the scan to the same neighbourhood used
  by the camera optimizer,
- aligned ESA WorldCover, only to reject class 80 (permanent water).

Outputs:
- all raw points that pass the terrain thresholds,
- final shoulder points after score ranking + minimum-spacing filtering.

Designed for QGIS 3.34+.
"""

import math

import numpy as np
from osgeo import gdal

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
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
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
)


gdal.UseExceptions()


class LesvosShoulderCandidateGenerator(QgsProcessingAlgorithm):
    DEM = "DEM"
    LANDCOVER = "LANDCOVER"
    TARGET_AREA = "TARGET_AREA"
    TARGET_BUFFER = "TARGET_BUFFER"
    TERRAIN_SEARCH_STEP = "TERRAIN_SEARCH_STEP"
    SHOULDER_PROBE_DISTANCE = "SHOULDER_PROBE_DISTANCE"
    SHOULDER_MAX_SITE_SLOPE = "SHOULDER_MAX_SITE_SLOPE"
    SHOULDER_MIN_DOWNHILL_SLOPE = "SHOULDER_MIN_DOWNHILL_SLOPE"
    SHOULDER_MAX_BACK_SLOPE = "SHOULDER_MAX_BACK_SLOPE"
    SHOULDER_MIN_BREAK = "SHOULDER_MIN_BREAK"
    SHOULDER_MIN_SPACING = "SHOULDER_MIN_SPACING"
    MAX_SHOULDER_CANDIDATES = "MAX_SHOULDER_CANDIDATES"
    MIN_ELEVATION = "MIN_ELEVATION"

    OUT_RAW = "OUT_RAW"
    OUT_SELECTED = "OUT_SELECTED"

    def tr(self, text):
        return QCoreApplication.translate("LesvosShoulderCandidateGenerator", text)

    def createInstance(self):
        return LesvosShoulderCandidateGenerator()

    def name(self):
        return "lesvos_shoulder_candidate_generator_v1"

    def displayName(self):
        return self.tr("Shoulder candidate generator – DEM only v1")

    def group(self):
        return self.tr("Πυρανίχνευση")

    def groupId(self):
        return "fire_detection"

    def shortHelpString(self):
        return self.tr(
            "Διαγνωστικό εργαλείο που παράγει μόνο terrain shoulder candidates από DEM. "
            "Δεν εκτελεί viewshed και δεν επιλέγει κάμερες. Σαρώνει το DEM με συγκεκριμένο βήμα και, "
            "για κάθε σημείο, εξετάζει 8 κατευθύνσεις. Ένα σημείο περνά όταν υπάρχει αρκετά έντονη "
            "πτώση προς μία κατεύθυνση, ηπιότερη αντίθετη/back κλίση, επαρκές terrain break και "
            "αποδεκτή τοπική κλίση εγκατάστασης. Στη συνέχεια τα σημεία ταξινομούνται με shoulder score "
            "και εφαρμόζεται minimum spacing. Αν δοθεί target polygon, η σάρωση περιορίζεται στο πραγματικό "
            "polygon + buffer. Αν δοθεί aligned WorldCover, απορρίπτεται η class 80 (νερό). "
            "Προτεινόμενο πρώτο test: search step=90 m, probe=150 m, site slope<=15°, downhill>=20°, "
            "back<=15°, break>=10°, spacing=300 m, max shoulders=500."
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
                self.tr("Προαιρετικά: ESA WorldCover aligned με το DEM (για απόρριψη water class 80)"),
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.TARGET_AREA,
                self.tr("Προαιρετικά: περιοχή στόχου (polygon)"),
                [QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TARGET_BUFFER,
                self.tr("Buffer γύρω από το target polygon για shoulder search (m)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=4000.0,
                minValue=0.0,
                maxValue=50000.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TERRAIN_SEARCH_STEP,
                self.tr("Βήμα σάρωσης terrain (m)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=90.0,
                minValue=30.0,
                maxValue=2000.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SHOULDER_PROBE_DISTANCE,
                self.tr("Απόσταση ελέγχου μπροστά/πίσω (m)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=150.0,
                minValue=30.0,
                maxValue=2000.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SHOULDER_MAX_SITE_SLOPE,
                self.tr("Μέγιστη τοπική κλίση θέσης shoulder (°)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=15.0,
                minValue=0.0,
                maxValue=60.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SHOULDER_MIN_DOWNHILL_SLOPE,
                self.tr("Ελάχιστη downhill κλίση (°)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=20.0,
                minValue=0.0,
                maxValue=80.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SHOULDER_MAX_BACK_SLOPE,
                self.tr("Μέγιστη back/uphill κλίση (°)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=15.0,
                minValue=-60.0,
                maxValue=60.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SHOULDER_MIN_BREAK,
                self.tr("Ελάχιστο terrain break = downhill - back (°)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
                minValue=0.0,
                maxValue=80.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SHOULDER_MIN_SPACING,
                self.tr("Ελάχιστη απόσταση μεταξύ τελικών shoulder candidates (m)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=300.0,
                minValue=0.0,
                maxValue=5000.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_SHOULDER_CANDIDATES,
                self.tr("Μέγιστος αριθμός τελικών shoulder candidates"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=500,
                minValue=1,
                maxValue=10000,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_ELEVATION,
                self.tr("Ελάχιστο υψόμετρο candidate (m)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=-500.0,
                maxValue=5000.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUT_RAW,
                self.tr("RAW shoulder points που περνούν τα terrain thresholds"),
                QgsProcessing.TypeVectorPoint,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUT_SELECTED,
                self.tr("Τελικά shoulder candidates μετά minimum spacing"),
                QgsProcessing.TypeVectorPoint,
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
                "Το script δεν υποστηρίζει rotated raster grid. Κάνε Align rasters ξανά."
            )
        col = int(math.floor((x - gt[0]) / gt[1]))
        row = int(math.floor((y - gt[3]) / gt[5]))
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return None
        return row, col

    @staticmethod
    def _site_slope_deg(row, col, dem, dem_valid, gt):
        """Estimate local terrain slope from neighbouring DEM cells."""
        rows, cols = dem.shape
        px = abs(float(gt[1]))
        py = abs(float(gt[5]))
        z0 = float(dem[row, col])

        def derivative_minus_plus(rm, cm, rp, cp, spacing):
            valid_m = 0 <= rm < rows and 0 <= cm < cols and bool(dem_valid[rm, cm])
            valid_p = 0 <= rp < rows and 0 <= cp < cols and bool(dem_valid[rp, cp])
            if valid_m and valid_p:
                return (float(dem[rp, cp]) - float(dem[rm, cm])) / (2.0 * spacing)
            if valid_p:
                return (float(dem[rp, cp]) - z0) / spacing
            if valid_m:
                return (z0 - float(dem[rm, cm])) / spacing
            return None

        dzdx = derivative_minus_plus(row, col - 1, row, col + 1, px)
        dzdy = derivative_minus_plus(row - 1, col, row + 1, col, py)
        if dzdx is None or dzdy is None:
            return None
        return math.degrees(math.atan(math.hypot(dzdx, dzdy)))

    @staticmethod
    def _output_fields():
        fields = QgsFields()
        fields.append(QgsField("rank", QVariant.Int))
        fields.append(QgsField("selected", QVariant.Int))
        fields.append(QgsField("score", QVariant.Double))
        fields.append(QgsField("elev_m", QVariant.Double))
        fields.append(QgsField("drop_az", QVariant.Double))
        fields.append(QgsField("down_deg", QVariant.Double))
        fields.append(QgsField("back_deg", QVariant.Double))
        fields.append(QgsField("break_deg", QVariant.Double))
        fields.append(QgsField("site_deg", QVariant.Double))
        fields.append(QgsField("row", QVariant.Int))
        fields.append(QgsField("col", QVariant.Int))
        return fields

    def _build_allowed_geometry(self, source, dem_crs, context, buffer_m):
        if source is None:
            return None

        transform = None
        if source.sourceCrs() != dem_crs:
            transform = QgsCoordinateTransform(source.sourceCrs(), dem_crs, context.transformContext())

        geoms = []
        for feature in source.getFeatures():
            geom = QgsGeometry(feature.geometry())
            if geom.isNull() or geom.isEmpty():
                continue
            if transform is not None:
                geom.transform(transform)
            geoms.append(geom)

        if not geoms:
            raise QgsProcessingException("Το target polygon δεν περιέχει έγκυρη γεωμετρία.")

        if len(geoms) == 1:
            allowed = geoms[0]
        else:
            allowed = QgsGeometry.unaryUnion(geoms)

        if allowed.isNull() or allowed.isEmpty():
            raise QgsProcessingException("Αποτυχία δημιουργίας ενιαίας target geometry.")

        if buffer_m > 0.0:
            allowed = allowed.buffer(float(buffer_m), 24)
            if allowed.isNull() or allowed.isEmpty():
                raise QgsProcessingException("Αποτυχία δημιουργίας target buffer.")

        return allowed

    def processAlgorithm(self, parameters, context, feedback):
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
        landcover_layer = self.parameterAsRasterLayer(parameters, self.LANDCOVER, context)
        target_source = self.parameterAsSource(parameters, self.TARGET_AREA, context)

        if dem_layer is None:
            raise QgsProcessingException("Δεν επιλέχθηκε DEM.")

        buffer_m = self.parameterAsDouble(parameters, self.TARGET_BUFFER, context)
        search_step = self.parameterAsDouble(parameters, self.TERRAIN_SEARCH_STEP, context)
        probe_distance = self.parameterAsDouble(parameters, self.SHOULDER_PROBE_DISTANCE, context)
        max_site_slope = self.parameterAsDouble(parameters, self.SHOULDER_MAX_SITE_SLOPE, context)
        min_downhill = self.parameterAsDouble(parameters, self.SHOULDER_MIN_DOWNHILL_SLOPE, context)
        max_back = self.parameterAsDouble(parameters, self.SHOULDER_MAX_BACK_SLOPE, context)
        min_break = self.parameterAsDouble(parameters, self.SHOULDER_MIN_BREAK, context)
        min_spacing = self.parameterAsDouble(parameters, self.SHOULDER_MIN_SPACING, context)
        max_candidates = self.parameterAsInt(parameters, self.MAX_SHOULDER_CANDIDATES, context)
        min_elevation = self.parameterAsDouble(parameters, self.MIN_ELEVATION, context)

        dem_ds = gdal.Open(dem_layer.source())
        if dem_ds is None:
            raise QgsProcessingException("Δεν μπόρεσε να ανοίξει το DEM με GDAL.")

        dem_band = dem_ds.GetRasterBand(1)
        dem = dem_band.ReadAsArray().astype(np.float32)
        gt = dem_ds.GetGeoTransform()
        rows, cols = dem.shape
        nodata = dem_band.GetNoDataValue()
        dem_valid = np.isfinite(dem)
        if nodata is not None:
            dem_valid &= ~np.isclose(dem, float(nodata))

        landcover = None
        lc_ds = None
        if landcover_layer is not None:
            lc_ds = gdal.Open(landcover_layer.source())
            if lc_ds is None:
                raise QgsProcessingException("Δεν μπόρεσε να ανοίξει το WorldCover με GDAL.")
            if not self._same_grid(dem_ds, lc_ds):
                raise QgsProcessingException(
                    "Το WorldCover δεν είναι ακριβώς aligned με το DEM. Κάνε Align rasters πριν το χρησιμοποιήσεις."
                )
            landcover = lc_ds.GetRasterBand(1).ReadAsArray()

        dem_crs = dem_layer.crs()
        if not dem_crs.isValid() or dem_crs.isGeographic():
            raise QgsProcessingException(
                "Το DEM πρέπει να είναι σε projected CRS με μονάδες μέτρα (π.χ. EPSG:32635)."
            )

        allowed_geometry = self._build_allowed_geometry(
            target_source, dem_crs, context, buffer_m
        )

        raster_min_x = float(gt[0])
        raster_max_x = float(gt[0] + cols * gt[1])
        raster_max_y = float(gt[3])
        raster_min_y = float(gt[3] + rows * gt[5])

        if allowed_geometry is None:
            min_x, max_x = raster_min_x, raster_max_x
            min_y, max_y = raster_min_y, raster_max_y
            feedback.pushInfo("Shoulder search: ολόκληρο το DEM extent.")
        else:
            bbox = allowed_geometry.boundingBox()
            min_x = max(raster_min_x, float(bbox.xMinimum()))
            max_x = min(raster_max_x, float(bbox.xMaximum()))
            min_y = max(raster_min_y, float(bbox.yMinimum()))
            max_y = min(raster_max_y, float(bbox.yMaximum()))
            feedback.pushInfo(
                f"Shoulder search: target polygon + buffer {buffer_m:.0f} m."
            )

        if min_x >= max_x or min_y >= max_y:
            raise QgsProcessingException("Το target/buffer δεν τέμνει το DEM.")

        x_values = np.arange(min_x + search_step / 2.0, max_x, search_step)
        y_values = np.arange(min_y + search_step / 2.0, max_y, search_step)
        directions = np.arange(0.0, 360.0, 45.0, dtype=np.float64)

        raw = []
        checked = 0
        seen_cells = set()
        total_rows_scan = max(1, len(y_values))

        for yi, y in enumerate(y_values):
            if feedback.isCanceled():
                break
            feedback.setProgress(70.0 * yi / total_rows_scan)

            for x in x_values:
                checked += 1
                rc = self._xy_to_row_col(float(x), float(y), gt, rows, cols)
                if rc is None:
                    continue
                row, col = rc

                # Several sampled XYs should not create duplicate candidates in one DEM cell.
                cell_key = (row, col)
                if cell_key in seen_cells:
                    continue
                seen_cells.add(cell_key)

                if not bool(dem_valid[row, col]):
                    continue
                if float(dem[row, col]) < min_elevation:
                    continue
                if landcover is not None and int(round(float(landcover[row, col]))) == 80:
                    continue

                x0 = float(gt[0] + (col + 0.5) * gt[1])
                y0 = float(gt[3] + (row + 0.5) * gt[5])

                if allowed_geometry is not None:
                    point_geom = QgsGeometry.fromPointXY(QgsPointXY(x0, y0))
                    if not (allowed_geometry.contains(point_geom) or allowed_geometry.touches(point_geom)):
                        continue

                site_slope = self._site_slope_deg(row, col, dem, dem_valid, gt)
                if site_slope is None or site_slope > max_site_slope:
                    continue

                z0 = float(dem[row, col])
                best = None

                for azimuth in directions:
                    rad = math.radians(float(azimuth))
                    dx = probe_distance * math.sin(rad)
                    dy = probe_distance * math.cos(rad)

                    front_rc = self._xy_to_row_col(x0 + dx, y0 + dy, gt, rows, cols)
                    back_rc = self._xy_to_row_col(x0 - dx, y0 - dy, gt, rows, cols)
                    if front_rc is None or back_rc is None:
                        continue

                    fr, fc = front_rc
                    br, bc = back_rc
                    if not bool(dem_valid[fr, fc]) or not bool(dem_valid[br, bc]):
                        continue

                    front_x = float(gt[0] + (fc + 0.5) * gt[1])
                    front_y = float(gt[3] + (fr + 0.5) * gt[5])
                    back_x = float(gt[0] + (bc + 0.5) * gt[1])
                    back_y = float(gt[3] + (br + 0.5) * gt[5])

                    front_dist = max(1e-9, math.hypot(front_x - x0, front_y - y0))
                    back_dist = max(1e-9, math.hypot(back_x - x0, back_y - y0))
                    z_front = float(dem[fr, fc])
                    z_back = float(dem[br, bc])

                    downhill = math.degrees(math.atan2(z0 - z_front, front_dist))
                    back_slope = math.degrees(math.atan2(z_back - z0, back_dist))
                    terrain_break = downhill - back_slope

                    if downhill + 1e-9 < min_downhill:
                        continue
                    if back_slope - 1e-9 > max_back:
                        continue
                    if terrain_break + 1e-9 < min_break:
                        continue

                    flat_bonus = max(0.0, max_site_slope - site_slope)
                    score = 0.50 * downhill + 0.35 * terrain_break + 0.15 * flat_bonus

                    if best is None or score > best[0]:
                        best = (
                            float(score),
                            float(azimuth),
                            float(downhill),
                            float(back_slope),
                            float(terrain_break),
                        )

                if best is None:
                    continue

                score, drop_az, downhill, back_slope, terrain_break = best
                raw.append(
                    {
                        "x": x0,
                        "y": y0,
                        "row": int(row),
                        "col": int(col),
                        "elev_m": z0,
                        "score": score,
                        "drop_az": drop_az,
                        "down_deg": downhill,
                        "back_deg": back_slope,
                        "break_deg": terrain_break,
                        "site_deg": float(site_slope),
                    }
                )

        if feedback.isCanceled():
            return {}

        raw.sort(key=lambda c: (-c["score"], -c["elev_m"]))

        selected = []
        if min_spacing <= 0.0:
            selected = raw[:max_candidates]
        else:
            min_spacing2 = float(min_spacing) ** 2
            for candidate in raw:
                if len(selected) >= max_candidates:
                    break
                keep = True
                for existing in selected:
                    dx = candidate["x"] - existing["x"]
                    dy = candidate["y"] - existing["y"]
                    if dx * dx + dy * dy < min_spacing2:
                        keep = False
                        break
                if keep:
                    selected.append(candidate)

        selected_key_to_rank = {
            (c["row"], c["col"]): rank
            for rank, c in enumerate(selected, start=1)
        }

        fields = self._output_fields()
        raw_sink, raw_dest = self.parameterAsSink(
            parameters,
            self.OUT_RAW,
            context,
            fields,
            QgsProcessing.TypeVectorPoint,
            dem_crs,
        )
        if raw_sink is None:
            raise QgsProcessingException("Δεν μπόρεσε να δημιουργηθεί το RAW output layer.")

        selected_sink, selected_dest = self.parameterAsSink(
            parameters,
            self.OUT_SELECTED,
            context,
            fields,
            QgsProcessing.TypeVectorPoint,
            dem_crs,
        )
        if selected_sink is None:
            raise QgsProcessingException("Δεν μπόρεσε να δημιουργηθεί το selected shoulder layer.")

        def add_feature(sink, candidate, rank, selected_flag):
            f = QgsFeature(fields)
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(candidate["x"], candidate["y"])))
            f.setAttributes([
                int(rank) if rank is not None else None,
                int(selected_flag),
                float(candidate["score"]),
                float(candidate["elev_m"]),
                float(candidate["drop_az"]),
                float(candidate["down_deg"]),
                float(candidate["back_deg"]),
                float(candidate["break_deg"]),
                float(candidate["site_deg"]),
                int(candidate["row"]),
                int(candidate["col"]),
            ])
            sink.addFeature(f, QgsFeatureSink.FastInsert)

        for idx, candidate in enumerate(raw, start=1):
            rank = selected_key_to_rank.get((candidate["row"], candidate["col"]))
            add_feature(raw_sink, candidate, rank, 1 if rank is not None else 0)

        for rank, candidate in enumerate(selected, start=1):
            add_feature(selected_sink, candidate, rank, 1)

        feedback.setProgress(100.0)
        feedback.pushInfo(
            f"Shoulder scan ολοκληρώθηκε: ελέγχθηκαν {checked} sampled positions, "
            f"{len(raw)} πέρασαν τα terrain thresholds, και κρατήθηκαν {len(selected)} "
            f"με minimum spacing {min_spacing:.0f} m."
        )
        feedback.pushInfo(
            "Το drop_az είναι η κατεύθυνση της ισχυρότερης πτώσης που έκανε το σημείο shoulder candidate — "
            "δεν είναι τελικό camera azimuth."
        )

        dem_ds = None
        lc_ds = None

        return {
            self.OUT_RAW: raw_dest,
            self.OUT_SELECTED: selected_dest,
        }
