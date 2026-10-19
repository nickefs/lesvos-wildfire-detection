# -*- coding: utf-8 -*-
"""
QGIS Processing Script: Lesvos Fire Camera Optimizer

Finds a greedy near-optimal set of camera locations and azimuths using:
- a projected DEM (metres),
- an aligned ESA WorldCover raster,
- an optional target-area polygon mask,
- automatic restriction of candidate cameras to the target polygon neighbourhood,
- GDAL viewshed,
- a horizontal field of view (default 180 degrees),
- land-cover risk weights,
- distance-dependent detection quality,
- hybrid candidate generation from a regular grid plus terrain shoulders/ridges,
- local uphill refinement around each initially selected camera,
- optional target polygon, actual buffered-polygon candidate filtering, binary-vs-quality selection objective, optional target coverage stopping rule, uncovered-target rasters, and forest-only diagnostics.

The algorithm first finds coarse/global camera positions and then searches nearby
higher terrain. If a target polygon is supplied, only weighted vegetation inside
that polygon contributes to the optimization and coverage percentages, while camera candidates
are generated only inside the actual buffered target polygon. A refined position is accepted only if the overall weighted
coverage remains within the user-defined allowed loss.

Designed for QGIS 3.34+ with the GDAL Processing provider enabled.

Version 7.1 suppresses verbose per-viewshed GDAL child logs and adds compact
viewshed/greedy progress messages for large candidate sets.
"""

import csv
import math
import os
import shutil
import tempfile
import time

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
    QgsProcessingFeedback,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterEnum,
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
    TARGET_AREA = "TARGET_AREA"
    CANDIDATES = "CANDIDATES"
    NUM_CAMERAS = "NUM_CAMERAS"
    TARGET_COVERAGE = "TARGET_COVERAGE"
    SELECTION_OBJECTIVE = "SELECTION_OBJECTIVE"
    FOV = "FOV"
    AZ_STEP = "AZ_STEP"
    OBSERVER_HEIGHT = "OBSERVER_HEIGHT"
    TARGET_HEIGHT = "TARGET_HEIGHT"
    MAX_DISTANCE = "MAX_DISTANCE"
    TARGET_CANDIDATE_BUFFER = "TARGET_CANDIDATE_BUFFER"
    GRID_SPACING = "GRID_SPACING"
    CANDIDATE_MODE = "CANDIDATE_MODE"
    TERRAIN_SEARCH_STEP = "TERRAIN_SEARCH_STEP"
    SHOULDER_PROBE_DISTANCE = "SHOULDER_PROBE_DISTANCE"
    SHOULDER_MAX_SITE_SLOPE = "SHOULDER_MAX_SITE_SLOPE"
    SHOULDER_MIN_DOWNHILL_SLOPE = "SHOULDER_MIN_DOWNHILL_SLOPE"
    SHOULDER_MAX_BACK_SLOPE = "SHOULDER_MAX_BACK_SLOPE"
    SHOULDER_MIN_BREAK = "SHOULDER_MIN_BREAK"
    SHOULDER_MIN_SPACING = "SHOULDER_MIN_SPACING"
    MAX_SHOULDER_CANDIDATES = "MAX_SHOULDER_CANDIDATES"
    MAX_CANDIDATES = "MAX_CANDIDATES"
    MIN_ELEVATION = "MIN_ELEVATION"
    REFINE_RADIUS = "REFINE_RADIUS"
    REFINE_STEP = "REFINE_STEP"
    REFINE_MAX_LOSS = "REFINE_MAX_LOSS"
    REFINE_MAX_CANDIDATES = "REFINE_MAX_CANDIDATES"

    OUT_CANDIDATES = "OUT_CANDIDATES"
    OUT_INITIAL_CAMERAS = "OUT_INITIAL_CAMERAS"
    OUT_CAMERAS = "OUT_CAMERAS"
    OUT_MOVES = "OUT_MOVES"
    OUT_FOV = "OUT_FOV"
    OUT_QUALITY = "OUT_QUALITY"
    OUT_COUNT = "OUT_COUNT"
    OUT_REMAINING_RISK = "OUT_REMAINING_RISK"
    OUT_UNCOVERED_TARGET = "OUT_UNCOVERED_TARGET"
    OUT_UNCOVERED_FOREST = "OUT_UNCOVERED_FOREST"
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
        return "lesvos_fire_camera_optimizer_hybrid_shoulders_v7_1_quietlogs"

    def displayName(self):
        return self.tr("Βελτιστοποίηση θέσεων καμερών – hybrid shoulders v7.1 quiet logs")

    def group(self):
        return self.tr("Πυρανίχνευση")

    def groupId(self):
        return "fire_detection"

    def shortHelpString(self):
        return self.tr(
            "Επιλέγει θέσεις και κατευθύνσεις καμερών με βάση DEM, WorldCover, "
            "viewshed, οριζόντιο FOV, απόσταση και προαιρετικό target polygon. "
            "Η v7 προσθέτει Candidate generation mode: Grid only, Shoulder/Terrain only ή Hybrid Grid + Shoulder. "
            "Τα shoulder candidates εντοπίζονται από το DEM ως σημεία με έντονη πτώση προς μία κατεύθυνση, "
            "ηπιότερο terrain στην αντίθετη και αποδεκτή τοπική κλίση. Το terrain score δεν επιλέγει κάμερα: "
            "απλώς αποφασίζει ποια terrain σημεία αξίζει να περάσουν στο κανονικό viewshed/greedy optimization. "
            "Υπάρχει επίσης objective Quality coverage ή Binary new coverage. Αν δοθεί polygon, οι υποψήφιες θέσεις καμερών περιορίζονται στο πραγματικό buffered polygon και το risk έξω από αυτό "
            "μηδενίζεται και τα βασικά bin_pct/eff_pct αναφέρονται στη σταθμισμένη positive-risk περιοχή στόχου, όχι σε ολόκληρη τη γεωμετρική επιφάνεια του polygon. "
            "Αν δοθεί στόχος κάλυψης > 0, ο αριθμός καμερών λειτουργεί ως μέγιστο όριο "
            "και η greedy επιλογή σταματά μόλις επιτευχθεί ο στόχος bin_pct. Με Binary new coverage, η επόμενη κάμερα επιλέγεται κυρίως από τη νέα ακάλυπτη περιοχή που προσθέτει. Το buffer υποψήφιων δεν εφαρμόζεται ως απλό bbox αλλά ως πραγματικό polygon buffer. Για πρώτη δοκιμή στη Λέσβο: "
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
                self.TARGET_AREA,
                self.tr("Προαιρετική περιοχή στόχου (polygon)"),
                [QgsProcessing.TypeVectorPolygon],
                optional=True,
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
                self.tr("Μέγιστος αριθμός καμερών (ή ακριβής αριθμός αν ο στόχος κάλυψης = 0)"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=10,
                minValue=1,
                maxValue=100,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TARGET_COVERAGE,
                self.tr("Στόχος κάλυψης bin_pct (%) - 0 = χρήση όλων των καμερών"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=90.0,
                minValue=0.0,
                maxValue=100.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.SELECTION_OBJECTIVE,
                self.tr("Κριτήριο greedy επιλογής"),
                options=[
                    self.tr("Quality coverage - μεγιστοποιεί weighted ποιότητα/eff_pct"),
                    self.tr("Binary new coverage - μεγιστοποιεί νέα ακάλυπτη περιοχή/bin_pct"),
                ],
                defaultValue=1,
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
                defaultValue=2.0,
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
                self.TARGET_CANDIDATE_BUFFER,
                self.tr("Buffer πραγματικού polygon για υποψήφιες θέσεις (m, 0 = Max Distance)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
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
            QgsProcessingParameterEnum(
                self.CANDIDATE_MODE,
                self.tr("Τρόπος δημιουργίας υποψήφιων θέσεων"),
                options=[
                    self.tr("Grid only"),
                    self.tr("Shoulder / terrain only"),
                    self.tr("Hybrid grid + shoulder"),
                ],
                defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TERRAIN_SEARCH_STEP,
                self.tr("Βήμα σάρωσης terrain για shoulders (m)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=90.0,
                minValue=30.0,
                maxValue=2000.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SHOULDER_PROBE_DISTANCE,
                self.tr("Απόσταση ελέγχου shoulder μπροστά/πίσω (m)"),
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
                self.tr("Ελάχιστη downhill κλίση για shoulder (°)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=20.0,
                minValue=0.0,
                maxValue=80.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SHOULDER_MAX_BACK_SLOPE,
                self.tr("Μέγιστη back/uphill κλίση για shoulder (°)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=15.0,
                minValue=-60.0,
                maxValue=60.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SHOULDER_MIN_BREAK,
                self.tr("Ελάχιστη διαφορά downhill-back slope / terrain break (°)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
                minValue=0.0,
                maxValue=80.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SHOULDER_MIN_SPACING,
                self.tr("Ελάχιστη απόσταση μεταξύ shoulder candidates (m)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=300.0,
                minValue=0.0,
                maxValue=5000.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_SHOULDER_CANDIDATES,
                self.tr("Μέγιστος αριθμός shoulder candidates"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=500,
                minValue=1,
                maxValue=5000,
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
                defaultValue=0.0,
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
                self.OUT_CANDIDATES,
                self.tr("Υποψήφιες θέσεις Grid / Shoulder που εξετάστηκαν"),
                QgsProcessing.TypeVectorPoint,
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
            QgsProcessingParameterRasterDestination(
                self.OUT_UNCOVERED_TARGET,
                self.tr("Ακάλυπτη περιοχή στόχου (1=ακάλυπτο, 0=καλυμμένο)"),
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUT_UNCOVERED_FOREST,
                self.tr("Ακάλυπτη δασική έκταση class 10 (1=ακάλυπτο, 0=καλυμμένο)"),
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

    def _build_target_area_mask(
        self,
        target_source,
        dem_crs,
        transform_context,
        gt,
        rows,
        cols,
        dem_valid,
        feedback,
        buffer_distance=0.0,
        label="Target polygon",
    ):
        """Rasterize a polygon layer to the DEM grid.

        With buffer_distance=0 this builds the actual target-area mask.
        With buffer_distance>0 this builds the allowed candidate-area mask, i.e.
        the real buffered polygon, not just a rectangular bbox around the polygon.
        Cameras may stand outside the original target polygon, but only inside
        this buffered polygon if a target area is supplied.
        """
        if target_source is None:
            return None

        mask = np.zeros((rows, cols), dtype=bool)
        source_crs = target_source.sourceCrs()
        coord_transform = None
        if source_crs.isValid() and source_crs != dem_crs:
            coord_transform = QgsCoordinateTransform(source_crs, dem_crs, transform_context)

        checked_features = 0
        masked_features = 0

        for feature in target_source.getFeatures():
            if feedback.isCanceled():
                break

            geometry = feature.geometry()
            if geometry is None or geometry.isEmpty():
                continue

            geom = QgsGeometry(geometry)
            if coord_transform is not None:
                geom.transform(coord_transform)

            if buffer_distance > 0.0:
                geom = geom.buffer(float(buffer_distance), 24)
                if geom is None or geom.isEmpty():
                    continue

            checked_features += 1
            bbox = geom.boundingBox()

            col_a = int(math.floor((bbox.xMinimum() - gt[0]) / gt[1])) - 1
            col_b = int(math.floor((bbox.xMaximum() - gt[0]) / gt[1])) + 1
            row_a = int(math.floor((bbox.yMinimum() - gt[3]) / gt[5])) - 1
            row_b = int(math.floor((bbox.yMaximum() - gt[3]) / gt[5])) + 1

            col0 = max(0, min(col_a, col_b))
            col1 = min(cols - 1, max(col_a, col_b))
            row0 = max(0, min(row_a, row_b))
            row1 = min(rows - 1, max(row_a, row_b))
            if row0 > row1 or col0 > col1:
                continue

            feature_has_pixels = False
            for row in range(row0, row1 + 1):
                y = float(gt[3] + (row + 0.5) * gt[5])
                for col in range(col0, col1 + 1):
                    if not dem_valid[row, col] or mask[row, col]:
                        continue
                    x = float(gt[0] + (col + 0.5) * gt[1])
                    point_geom = QgsGeometry.fromPointXY(QgsPointXY(x, y))
                    if geom.contains(point_geom) or geom.touches(point_geom):
                        mask[row, col] = True
                        feature_has_pixels = True

            if feature_has_pixels:
                masked_features += 1

        if checked_features == 0:
            raise QgsProcessingException(
                f"{label}: το polygon layer δεν περιέχει έγκυρα πολύγωνα."
            )
        if not np.any(mask & dem_valid):
            raise QgsProcessingException(
                f"{label}: το polygon δεν επικαλύπτει το DEM ή βρίσκεται σε λάθος CRS."
            )

        feedback.pushInfo(
            f"{label}: χρησιμοποιήθηκαν {masked_features}/{checked_features} πολύγωνα, "
            f"pixels μέσα στο DEM: {int(np.count_nonzero(mask & dem_valid))}."
        )
        return mask

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
        candidate_bounds=None,
        candidate_allowed_mask=None,
        candidate_type="GRID",
        apply_max_limit=True,
    ):
        candidates = []
        seen_cells = set()

        def try_add(x, y):
            if candidate_bounds is not None:
                min_x_allowed, max_x_allowed, min_y_allowed, max_y_allowed = candidate_bounds
                if (
                    x < min_x_allowed
                    or x > max_x_allowed
                    or y < min_y_allowed
                    or y > max_y_allowed
                ):
                    return
            rc = self._xy_to_row_col(x, y, gt, rows, cols)
            if rc is None:
                return
            row, col = rc
            if candidate_allowed_mask is not None and not bool(candidate_allowed_mask[row, col]):
                return
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
            candidates.append({
                "x": float(x),
                "y": float(y),
                "row": row,
                "col": col,
                "candidate_type": str(candidate_type),
                "shoulder_score": None,
                "shoulder_azimuth_deg": None,
                "downhill_slope_deg": None,
                "back_slope_deg": None,
                "terrain_break_deg": None,
                "site_slope_deg": None,
            })

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

        if apply_max_limit and len(candidates) > max_candidates:
            # Deterministic, spatially distributed down-sampling of the regular list.
            keep_idx = np.linspace(0, len(candidates) - 1, max_candidates, dtype=int)
            candidates = [candidates[int(i)] for i in keep_idx]
            feedback.pushInfo(
                f"Τα υποψήφια σημεία περιορίστηκαν σε {max_candidates} για να παραμείνει πρακτικός ο χρόνος εκτέλεσης."
            )

        return candidates

    @staticmethod
    def _site_slope_deg(row, col, dem, dem_valid, gt):
        """Estimate local terrain slope from central/one-sided DEM differences."""
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
        # Raster row increases southward; sign does not matter for slope magnitude.
        dzdy = derivative_minus_plus(row - 1, col, row + 1, col, py)
        if dzdx is None or dzdy is None:
            return None
        return math.degrees(math.atan(math.hypot(dzdx, dzdy)))

    def _build_shoulder_candidates(
        self,
        gt,
        rows,
        cols,
        dem,
        dem_valid,
        landcover,
        min_elevation,
        search_step,
        probe_distance,
        max_site_slope,
        min_downhill_slope,
        max_back_slope,
        min_break,
        min_spacing,
        max_candidates,
        feedback,
        candidate_bounds=None,
        candidate_allowed_mask=None,
    ):
        """Detect terrain shoulder/ridge candidates from the DEM.

        A point is a terrain candidate when at least one of 8 directions has:
        - a sufficiently steep drop in front of the point,
        - a milder opposite/back slope,
        - a sufficient terrain-break difference,
        - and an acceptable local site slope.

        Terrain score only ranks candidate points; it never replaces viewshed scoring.
        """
        if search_step <= 0.0 or probe_distance <= 0.0 or max_candidates <= 0:
            return []

        raster_min_x = float(gt[0])
        raster_max_x = float(gt[0] + cols * gt[1])
        raster_max_y = float(gt[3])
        raster_min_y = float(gt[3] + rows * gt[5])
        if candidate_bounds is None:
            min_x, max_x = raster_min_x, raster_max_x
            min_y, max_y = raster_min_y, raster_max_y
        else:
            min_x, max_x, min_y, max_y = candidate_bounds
            min_x = max(min_x, raster_min_x)
            max_x = min(max_x, raster_max_x)
            min_y = max(min_y, raster_min_y)
            max_y = min(max_y, raster_max_y)

        x_values = np.arange(min_x + search_step / 2.0, max_x, search_step)
        y_values = np.arange(min_y + search_step / 2.0, max_y, search_step)
        directions = np.arange(0.0, 360.0, 45.0, dtype=np.float64)
        raw = []
        checked = 0

        for y in y_values:
            if feedback.isCanceled():
                break
            for x in x_values:
                checked += 1
                rc = self._xy_to_row_col(float(x), float(y), gt, rows, cols)
                if rc is None:
                    continue
                row, col = rc
                if not dem_valid[row, col]:
                    continue
                if candidate_allowed_mask is not None and not bool(candidate_allowed_mask[row, col]):
                    continue
                if float(dem[row, col]) < min_elevation:
                    continue
                if int(round(float(landcover[row, col]))) == 80:
                    continue

                site_slope = self._site_slope_deg(row, col, dem, dem_valid, gt)
                if site_slope is None or site_slope > max_site_slope:
                    continue

                # Use DEM cell centre as the actual candidate coordinate.
                x0 = float(gt[0] + (col + 0.5) * gt[1])
                y0 = float(gt[3] + (row + 0.5) * gt[5])
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
                    if not dem_valid[fr, fc] or not dem_valid[br, bc]:
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

                    if downhill + 1e-9 < min_downhill_slope:
                        continue
                    if back_slope - 1e-9 > max_back_slope:
                        continue
                    if terrain_break + 1e-9 < min_break:
                        continue

                    # Ranking score: strong forward drop and break are rewarded;
                    # flatter installation terrain gets a smaller bonus.
                    flat_bonus = max(0.0, max_site_slope - site_slope)
                    score = 0.50 * downhill + 0.35 * terrain_break + 0.15 * flat_bonus
                    if best is None or score > best[0]:
                        best = (
                            score,
                            float(azimuth),
                            downhill,
                            back_slope,
                            terrain_break,
                        )

                if best is None:
                    continue
                score, shoulder_azimuth, downhill, back_slope, terrain_break = best
                raw.append(
                    {
                        "x": x0,
                        "y": y0,
                        "row": row,
                        "col": col,
                        "candidate_type": "SHOULDER",
                        "shoulder_score": float(score),
                        "shoulder_azimuth_deg": float(shoulder_azimuth),
                        "downhill_slope_deg": float(downhill),
                        "back_slope_deg": float(back_slope),
                        "terrain_break_deg": float(terrain_break),
                        "site_slope_deg": float(site_slope),
                    }
                )

        raw.sort(key=lambda c: (-c["shoulder_score"], -float(dem[c["row"], c["col"]])))
        if min_spacing <= 0.0:
            selected = raw[:max_candidates]
        else:
            selected = []
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

        feedback.pushInfo(
            f"Terrain shoulder scan: ελέγχθηκαν {checked} σημεία, "
            f"{len(raw)} πέρασαν τα terrain thresholds και κρατήθηκαν {len(selected)} "
            f"με minimum spacing {min_spacing:.0f} m."
        )
        return selected

    @staticmethod
    def _merge_hybrid_candidates(grid_candidates, shoulder_candidates, max_candidates, feedback):
        """Merge candidates, preferring shoulder metadata on exact raster-cell duplicates."""
        shoulder_by_cell = {(c["row"], c["col"]): c for c in shoulder_candidates}
        grid_unique = [
            c for c in grid_candidates if (c["row"], c["col"]) not in shoulder_by_cell
        ]
        shoulders = list(shoulder_candidates)

        if len(shoulders) >= max_candidates:
            combined = shoulders[:max_candidates]
        else:
            remaining_slots = max_candidates - len(shoulders)
            if len(grid_unique) > remaining_slots:
                keep_idx = np.linspace(0, len(grid_unique) - 1, remaining_slots, dtype=int)
                grid_unique = [grid_unique[int(i)] for i in keep_idx]
            combined = shoulders + grid_unique

        feedback.pushInfo(
            f"Hybrid candidates μετά το merge: {len(shoulders)} shoulder + "
            f"{len(combined) - len(shoulders)} grid = {len(combined)} συνολικά."
        )
        return combined

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
        candidate_allowed_mask=None,
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
                if candidate_allowed_mask is not None and not bool(candidate_allowed_mask[row, col]):
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
                feedback=QgsProcessingFeedback(False),
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
        target_area_source = self.parameterAsSource(parameters, self.TARGET_AREA, context)
        candidate_source = self.parameterAsSource(parameters, self.CANDIDATES, context)

        if dem_layer is None or landcover_layer is None:
            raise QgsProcessingException("Πρέπει να επιλεγούν DEM και landcover raster.")

        num_cameras = self.parameterAsInt(parameters, self.NUM_CAMERAS, context)
        target_coverage = self.parameterAsDouble(parameters, self.TARGET_COVERAGE, context)
        selection_objective = self.parameterAsEnum(parameters, self.SELECTION_OBJECTIVE, context)
        use_binary_objective = int(selection_objective) == 1
        selection_objective_label = "binary_new_coverage" if use_binary_objective else "quality_coverage"
        fov = self.parameterAsDouble(parameters, self.FOV, context)
        az_step = self.parameterAsDouble(parameters, self.AZ_STEP, context)
        observer_height = self.parameterAsDouble(parameters, self.OBSERVER_HEIGHT, context)
        target_height = self.parameterAsDouble(parameters, self.TARGET_HEIGHT, context)
        max_distance = self.parameterAsDouble(parameters, self.MAX_DISTANCE, context)
        target_candidate_buffer = self.parameterAsDouble(parameters, self.TARGET_CANDIDATE_BUFFER, context)
        grid_spacing = self.parameterAsDouble(parameters, self.GRID_SPACING, context)
        candidate_mode = self.parameterAsEnum(parameters, self.CANDIDATE_MODE, context)
        candidate_mode_labels = {0: "grid_only", 1: "shoulder_only", 2: "hybrid_grid_shoulder"}
        candidate_mode_label = candidate_mode_labels.get(int(candidate_mode), "hybrid_grid_shoulder")
        terrain_search_step = self.parameterAsDouble(parameters, self.TERRAIN_SEARCH_STEP, context)
        shoulder_probe_distance = self.parameterAsDouble(parameters, self.SHOULDER_PROBE_DISTANCE, context)
        shoulder_max_site_slope = self.parameterAsDouble(parameters, self.SHOULDER_MAX_SITE_SLOPE, context)
        shoulder_min_downhill_slope = self.parameterAsDouble(parameters, self.SHOULDER_MIN_DOWNHILL_SLOPE, context)
        shoulder_max_back_slope = self.parameterAsDouble(parameters, self.SHOULDER_MAX_BACK_SLOPE, context)
        shoulder_min_break = self.parameterAsDouble(parameters, self.SHOULDER_MIN_BREAK, context)
        shoulder_min_spacing = self.parameterAsDouble(parameters, self.SHOULDER_MIN_SPACING, context)
        max_shoulder_candidates = self.parameterAsInt(parameters, self.MAX_SHOULDER_CANDIDATES, context)
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

        target_mask = None
        target_polygon_pixels = 0
        if target_area_source is not None:
            feedback.pushInfo("Εφαρμογή προαιρετικού target polygon: μηδενισμός risk εκτός περιοχής στόχου…")
            target_mask = self._build_target_area_mask(
                target_area_source,
                dem_crs,
                context.transformContext(),
                gt,
                rows,
                cols,
                dem_valid,
                feedback,
                buffer_distance=0.0,
                label="Target polygon",
            )
            target_polygon_pixels = int(np.count_nonzero(target_mask & dem_valid))
            risk[~target_mask] = 0.0

        candidate_bounds = None
        candidate_allowed_mask = None
        candidate_search_buffer = 0.0
        candidate_allowed_pixels = 0
        if target_mask is not None:
            # Important speed optimisation: a camera farther than the useful
            # detection range from the target polygon cannot add coverage there.
            # Therefore, when a target polygon is supplied, do not run expensive
            # viewsheds for automatic candidates across the whole island.
            # Version 5 uses the actual buffered polygon, not just bbox + buffer.
            candidate_search_buffer = float(target_candidate_buffer)
            if candidate_search_buffer <= 0.0:
                candidate_search_buffer = float(max_distance)

            candidate_allowed_mask = self._build_target_area_mask(
                target_area_source,
                dem_crs,
                context.transformContext(),
                gt,
                rows,
                cols,
                dem_valid,
                feedback,
                buffer_distance=candidate_search_buffer,
                label="Buffered candidate polygon",
            )
            candidate_allowed_pixels = int(np.count_nonzero(candidate_allowed_mask & dem_valid))

            allowed_rows, allowed_cols = np.nonzero(candidate_allowed_mask & dem_valid)
            if allowed_rows.size == 0 or allowed_cols.size == 0:
                raise QgsProcessingException(
                    "Το buffered target polygon δεν έχει έγκυρα DEM pixels για δημιουργία υποψήφιων θέσεων."
                )

            col_min = int(allowed_cols.min())
            col_max = int(allowed_cols.max())
            row_min = int(allowed_rows.min())
            row_max = int(allowed_rows.max())
            x_min = float(gt[0] + col_min * gt[1])
            x_max = float(gt[0] + (col_max + 1) * gt[1])
            y_top = float(gt[3] + row_min * gt[5])
            y_bottom = float(gt[3] + (row_max + 1) * gt[5])
            y_min = min(y_bottom, y_top)
            y_max = max(y_bottom, y_top)

            # This bbox is only used to avoid scanning the whole raster grid.
            # The real filter is candidate_allowed_mask, i.e. the true buffered polygon.
            candidate_bounds = (x_min, x_max, y_min, y_max)
            feedback.pushInfo(
                "Target polygon: οι υποψήφιες θέσεις θα περιοριστούν στο πραγματικό "
                f"buffered polygon + {candidate_search_buffer:.0f} m, όχι στο απλό bbox."
            )

        risk_flat = risk.ravel()
        target_risk_pixels = int(np.count_nonzero(risk_flat > 0))
        total_risk = float(risk_flat.sum(dtype=np.float64))
        if total_risk <= 0:
            raise QgsProcessingException(
                "Ο χάρτης βλάστησης/target area δεν περιέχει αναγνωρίσιμες WorldCover τιμές με θετικό βάρος. Έλεγξε το landcover, την ευθυγράμμιση ή το polygon."
            )

        # Diagnostic masks used at the end for plain area percentages.
        # IMPORTANT: the optimizer's bin_pct/eff_pct still use risk weights.
        target_positive_mask_2d = (risk > 0.0) & dem_valid
        forest_mask_2d = (lc_int == 10) & lc_valid & dem_valid
        if target_mask is not None:
            forest_mask_2d &= target_mask
        target_positive_pixels = int(np.count_nonzero(target_positive_mask_2d))
        forest_total_pixels = int(np.count_nonzero(forest_mask_2d))
        pixel_area_ha = abs(float(gt[1]) * float(gt[5])) / 10000.0

        use_target_coverage = target_coverage > 0.0
        target_coverage_reached_initial = False
        target_coverage_reached_rank = 0
        if use_target_coverage:
            feedback.pushInfo(
                f"Λειτουργία στόχου κάλυψης: θα χρησιμοποιηθούν έως {num_cameras} κάμερες "
                f"και η επιλογή θα σταματήσει όταν το bin_pct φτάσει {target_coverage:.2f}%."
            )
        else:
            feedback.pushInfo(
                f"Λειτουργία σταθερού πλήθους: θα επιλεγούν έως {num_cameras} κάμερες."
            )

        if use_binary_objective:
            feedback.pushInfo(
                "Κριτήριο greedy επιλογής: Binary new coverage — η επόμενη κάμερα "
                "διαλέγεται κυρίως από το νέο ακάλυπτο weighted risk που προσθέτει."
            )
        else:
            feedback.pushInfo(
                "Κριτήριο greedy επιλογής: Quality coverage — η επόμενη κάμερα "
                "διαλέγεται από τη βελτίωση weighted ποιότητας/eff_pct."
            )

        feedback.pushInfo("Δημιουργία υποψήφιων θέσεων…")
        grid_candidate_count_generated = 0
        shoulder_candidate_count_generated = 0

        if candidate_source is not None:
            # Backwards-compatible behaviour: an explicitly supplied point layer is
            # treated as the complete candidate set. Terrain/grid generation is skipped.
            candidate_mode_label = "supplied_points"
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
                candidate_bounds=candidate_bounds,
                candidate_allowed_mask=candidate_allowed_mask,
                candidate_type="SUPPLIED",
                apply_max_limit=True,
            )
            feedback.pushInfo(
                "Δόθηκε layer υποψήφιων σημείων: χρησιμοποιούνται μόνο τα supplied points "
                "και αγνοείται το Grid/Shoulder generation mode."
            )
        elif int(candidate_mode) == 0:
            candidates = self._build_candidates(
                None,
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
                candidate_bounds=candidate_bounds,
                candidate_allowed_mask=candidate_allowed_mask,
                candidate_type="GRID",
                apply_max_limit=True,
            )
            grid_candidate_count_generated = len(candidates)
            feedback.pushInfo(f"Candidate mode: Grid only — {len(candidates)} grid candidates.")
        elif int(candidate_mode) == 1:
            candidates = self._build_shoulder_candidates(
                gt, rows, cols, dem, dem_valid, landcover, min_elevation,
                terrain_search_step, shoulder_probe_distance,
                shoulder_max_site_slope, shoulder_min_downhill_slope,
                shoulder_max_back_slope, shoulder_min_break,
                shoulder_min_spacing, min(max_shoulder_candidates, max_candidates),
                feedback, candidate_bounds=candidate_bounds,
                candidate_allowed_mask=candidate_allowed_mask,
            )
            shoulder_candidate_count_generated = len(candidates)
            if not candidates:
                raise QgsProcessingException(
                    "Δεν βρέθηκαν shoulder candidates με τα τρέχοντα terrain thresholds. "
                    "Χαλάρωσε τα slope thresholds ή χρησιμοποίησε Hybrid/Grid mode."
                )
            feedback.pushInfo(f"Candidate mode: Shoulder only — {len(candidates)} terrain candidates.")
        else:
            grid_candidates = self._build_candidates(
                None,
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
                candidate_bounds=candidate_bounds,
                candidate_allowed_mask=candidate_allowed_mask,
                candidate_type="GRID",
                apply_max_limit=False,
            )
            shoulder_candidates = self._build_shoulder_candidates(
                gt, rows, cols, dem, dem_valid, landcover, min_elevation,
                terrain_search_step, shoulder_probe_distance,
                shoulder_max_site_slope, shoulder_min_downhill_slope,
                shoulder_max_back_slope, shoulder_min_break,
                shoulder_min_spacing, max_shoulder_candidates,
                feedback, candidate_bounds=candidate_bounds,
                candidate_allowed_mask=candidate_allowed_mask,
            )
            grid_candidate_count_generated = len(grid_candidates)
            shoulder_candidate_count_generated = len(shoulder_candidates)
            candidates = self._merge_hybrid_candidates(
                grid_candidates, shoulder_candidates, max_candidates, feedback
            )

        if not candidates:
            raise QgsProcessingException("Δεν δημιουργήθηκε κανένα έγκυρο candidate point.")

        final_grid_candidate_count = sum(1 for c in candidates if c.get("candidate_type") == "GRID")
        final_shoulder_candidate_count = sum(1 for c in candidates if c.get("candidate_type") == "SHOULDER")
        final_supplied_candidate_count = sum(1 for c in candidates if c.get("candidate_type") == "SUPPLIED")
        feedback.pushInfo(
            f"Έγκυρες υποψήφιες θέσεις για viewshed: {len(candidates)} "
            f"(GRID={final_grid_candidate_count}, SHOULDER={final_shoulder_candidate_count}, "
            f"SUPPLIED={final_supplied_candidate_count})."
        )

        # Diagnostic candidate layer: lets the user inspect where the terrain detector
        # placed shoulder points before any viewshed/greedy selection.
        candidate_fields = QgsFields()
        candidate_fields.append(QgsField("cand_id", QVariant.Int))
        candidate_fields.append(QgsField("cand_type", QVariant.String))
        candidate_fields.append(QgsField("elev_m", QVariant.Double))
        candidate_fields.append(QgsField("sh_score", QVariant.Double))
        candidate_fields.append(QgsField("sh_az", QVariant.Double))
        candidate_fields.append(QgsField("down_slp", QVariant.Double))
        candidate_fields.append(QgsField("back_slp", QVariant.Double))
        candidate_fields.append(QgsField("break_deg", QVariant.Double))
        candidate_fields.append(QgsField("site_slp", QVariant.Double))
        candidate_sink, candidate_dest = self.parameterAsSink(
            parameters,
            self.OUT_CANDIDATES,
            context,
            candidate_fields,
            QgsWkbTypes.Point,
            dem_crs,
        )
        if candidate_sink is None:
            raise QgsProcessingException("Δεν μπόρεσε να δημιουργηθεί το layer υποψήφιων θέσεων.")
        for candidate_id, candidate in enumerate(candidates, start=1):
            feature = QgsFeature(candidate_fields)
            feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(candidate["x"], candidate["y"])))
            feature.setAttributes([
                candidate_id,
                candidate.get("candidate_type", "GRID"),
                float(dem[candidate["row"], candidate["col"]]),
                candidate.get("shoulder_score"),
                candidate.get("shoulder_azimuth_deg"),
                candidate.get("downhill_slope_deg"),
                candidate.get("back_slope_deg"),
                candidate.get("terrain_break_deg"),
                candidate.get("site_slope_deg"),
            ])
            candidate_sink.addFeature(feature, QgsFeatureSink.FastInsert)

        alg = QgsApplication.processingRegistry().algorithmById("gdal:viewshed")
        if alg is None:
            raise QgsProcessingException(
                "Δεν βρέθηκε το gdal:viewshed. Ενεργοποίησε τον GDAL provider στο Processing."
            )
        parameter_names = {definition.name() for definition in alg.parameterDefinitions()}

        temp_dir = tempfile.mkdtemp(prefix="firecam_viewshed_")
        cache = []
        usable_candidates = []

        # Use a silent child feedback object for GDAL viewshed calls. This prevents
        # QGIS from printing the full GDAL command/result block for every single
        # candidate, while the parent feedback still reports compact progress.
        quiet_viewshed_feedback = QgsProcessingFeedback(False)
        viewshed_log_step = max(25, int(math.ceil(len(candidates) / 20.0)))
        viewshed_started = time.monotonic()

        try:
            feedback.pushInfo(
                f"Viewshed phase: {len(candidates)} candidates. "
                f"GDAL child logs are suppressed; compact progress will be shown instead."
            )
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
                        feedback=quiet_viewshed_feedback,
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

                completed_viewsheds = index + 1
                if (
                    completed_viewsheds == 1
                    or completed_viewsheds % viewshed_log_step == 0
                    or completed_viewsheds == len(candidates)
                ):
                    elapsed = time.monotonic() - viewshed_started
                    feedback.pushInfo(
                        f"Viewshed progress: {completed_viewsheds}/{len(candidates)} "
                        f"({100.0 * completed_viewsheds / max(1, len(candidates)):.1f}%), "
                        f"usable={len(cache)}, elapsed={elapsed / 60.0:.1f} min."
                    )

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

            feedback.pushInfo(
                f"Greedy επιλογή θέσεων και κατευθύνσεων: {len(cache)} usable candidates, "
                f"{len(azimuths)} azimuths ανά candidate."
            )
            # The GUI progress bar used to barely move when MAX_CAMERAS was large
            # (e.g. 100) even though a target-coverage run usually stops much earlier.
            # Use a practical 20-camera progress budget and explicit textual progress.
            greedy_progress_camera_budget = max(1, min(int(num_cameras), 20))

            for rank in range(1, num_cameras + 1):
                if feedback.isCanceled():
                    break
                best = None
                best_gain = 0.0
                rank_candidate_count = len(remaining)
                greedy_log_step = max(50, int(math.ceil(rank_candidate_count / 10.0)))
                rank_started = time.monotonic()
                feedback.pushInfo(
                    f"Greedy camera {rank}: checking {rank_candidate_count} candidates "
                    f"× {len(azimuths)} azimuths…"
                )

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
                        selected_quality = quality[mask]
                        selected_existing = existing[mask]

                        if use_binary_objective:
                            # Main objective for minimising camera count:
                            # count only target/risk pixels that were previously completely uncovered.
                            new_pixels = coverage_count[selected_indices] == 0
                            if not np.any(new_pixels):
                                continue
                            binary_gain = float(
                                np.sum(
                                    risk_flat[selected_indices[new_pixels]],
                                    dtype=np.float64,
                                )
                            )

                            # Tiny tie-breaker: if two orientations add almost the same new area,
                            # prefer the one with better distance/quality on those new pixels.
                            quality_tiebreak = float(
                                np.sum(
                                    risk_flat[selected_indices[new_pixels]] * selected_quality[new_pixels],
                                    dtype=np.float64,
                                )
                            )
                            gain = binary_gain + 1e-6 * quality_tiebreak
                        else:
                            # Original v5 objective: reward quality improvement, even when
                            # the pixel was already covered by another camera.
                            improvement = selected_quality - selected_existing
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

                    completed_candidates = position + 1
                    if (
                        completed_candidates % greedy_log_step == 0
                        or completed_candidates == rank_candidate_count
                    ):
                        elapsed_rank = time.monotonic() - rank_started
                        feedback.pushInfo(
                            f"Greedy camera {rank}: {completed_candidates}/{rank_candidate_count} "
                            f"candidates checked ({100.0 * completed_candidates / max(1, rank_candidate_count):.0f}%), "
                            f"elapsed={elapsed_rank / 60.0:.1f} min."
                        )

                    # More visible phase progress, even when the user set a very large
                    # maximum camera count. The exact textual counters above remain the
                    # authoritative progress indicator.
                    if position % 20 == 0 or completed_candidates == rank_candidate_count:
                        camera_slot = min(rank - 1, greedy_progress_camera_budget - 1)
                        fraction_in_rank = completed_candidates / max(1, rank_candidate_count)
                        progress_fraction = min(
                            1.0,
                            (camera_slot + fraction_in_rank) / greedy_progress_camera_budget,
                        )
                        feedback.setProgress(66.0 + 24.0 * progress_fraction)

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
                        "candidate_type": usable_candidates[candidate_index].get("candidate_type", "GRID"),
                        "shoulder_score": usable_candidates[candidate_index].get("shoulder_score"),
                        "shoulder_azimuth_deg": usable_candidates[candidate_index].get("shoulder_azimuth_deg"),
                        "downhill_slope_deg": usable_candidates[candidate_index].get("downhill_slope_deg"),
                        "back_slope_deg": usable_candidates[candidate_index].get("back_slope_deg"),
                        "terrain_break_deg": usable_candidates[candidate_index].get("terrain_break_deg"),
                        "site_slope_deg": usable_candidates[candidate_index].get("site_slope_deg"),
                        "indices": indices.copy(),
                        "quality": quality.copy(),
                    }
                )
                feedback.pushInfo(
                    f"{rank}. θέση ({selected[-1]['x']:.1f}, {selected[-1]['y']:.1f}), "
                    f"azimuth {azimuth:.0f}°, type {selected[-1]['candidate_type']}, "
                    f"eff_pct {effective_pct:.2f}%, bin_pct {binary_pct:.2f}%"
                    f", objective_gain {best_gain:.2f}"
                )

                if use_target_coverage and binary_pct + 1e-9 >= target_coverage:
                    target_coverage_reached_initial = True
                    target_coverage_reached_rank = rank
                    feedback.pushInfo(
                        f"Ο στόχος κάλυψης {target_coverage:.2f}% επιτεύχθηκε με {rank} κάμερες "
                        f"(bin_pct={binary_pct:.2f}%). Η greedy επιλογή σταματά εδώ."
                    )
                    break

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
                        "candidate_type": camera.get("candidate_type", "GRID"),
                        "shoulder_score": camera.get("shoulder_score"),
                        "shoulder_azimuth_deg": camera.get("shoulder_azimuth_deg"),
                        "downhill_slope_deg": camera.get("downhill_slope_deg"),
                        "back_slope_deg": camera.get("back_slope_deg"),
                        "terrain_break_deg": camera.get("terrain_break_deg"),
                        "site_slope_deg": camera.get("site_slope_deg"),
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

            initial_system_quality, initial_system_count = self._compose_coverage(
                refined_selected, rows * cols
            )
            initial_system_score = float(
                np.sum(risk_flat * initial_system_quality, dtype=np.float64)
            )
            initial_binary_after_greedy = 100.0 * float(
                np.sum(risk_flat[initial_system_count > 0], dtype=np.float64) / total_risk
            )
            required_binary_after_refine = 0.0
            if use_target_coverage and target_coverage_reached_initial:
                # If the greedy phase reached the requested coverage, local uphill moves
                # are not allowed to push the final binary weighted coverage below it.
                required_binary_after_refine = float(target_coverage)
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
                        candidate_allowed_mask=candidate_allowed_mask,
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

                        if required_binary_after_refine > 0.0:
                            test_cameras = list(refined_selected)
                            test_camera = dict(current_camera)
                            test_camera["indices"] = orientation["indices"]
                            test_camera["quality"] = orientation["quality"]
                            test_cameras[camera_pos] = test_camera
                            _, test_count = self._compose_coverage(test_cameras, rows * cols)
                            test_binary_pct = 100.0 * float(
                                np.sum(risk_flat[test_count > 0], dtype=np.float64) / total_risk
                            )
                            if test_binary_pct + 1e-9 < required_binary_after_refine:
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
            initial_fields.append(QgsField("cand_type", QVariant.String))
            initial_fields.append(QgsField("sh_score", QVariant.Double))
            initial_fields.append(QgsField("sh_az", QVariant.Double))
            initial_fields.append(QgsField("down_slp", QVariant.Double))
            initial_fields.append(QgsField("back_slp", QVariant.Double))
            initial_fields.append(QgsField("break_deg", QVariant.Double))
            initial_fields.append(QgsField("site_slp", QVariant.Double))

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
            final_fields.append(QgsField("cand_type", QVariant.String))
            final_fields.append(QgsField("sh_score", QVariant.Double))
            final_fields.append(QgsField("sh_az", QVariant.Double))
            final_fields.append(QgsField("down_slp", QVariant.Double))
            final_fields.append(QgsField("back_slp", QVariant.Double))
            final_fields.append(QgsField("break_deg", QVariant.Double))
            final_fields.append(QgsField("site_slp", QVariant.Double))

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
                        initial.get("candidate_type", "GRID"),
                        initial.get("shoulder_score"),
                        initial.get("shoulder_azimuth_deg"),
                        initial.get("downhill_slope_deg"),
                        initial.get("back_slope_deg"),
                        initial.get("terrain_break_deg"),
                        initial.get("site_slope_deg"),
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
                    camera.get("candidate_type", "GRID"),
                    camera.get("shoulder_score"),
                    camera.get("shoulder_azimuth_deg"),
                    camera.get("downhill_slope_deg"),
                    camera.get("back_slope_deg"),
                    camera.get("terrain_break_deg"),
                    camera.get("site_slope_deg"),
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
            output_uncovered_target = self.parameterAsOutputLayer(
                parameters, self.OUT_UNCOVERED_TARGET, context
            )
            output_uncovered_forest = self.parameterAsOutputLayer(
                parameters, self.OUT_UNCOVERED_FOREST, context
            )
            summary_path = self.parameterAsFileOutput(parameters, self.OUT_SUMMARY, context)

            quality_2d = current_quality.reshape((rows, cols)).copy()
            count_2d = coverage_count.reshape((rows, cols)).copy()
            remaining_risk = risk * (1.0 - quality_2d)

            # Binary diagnostics. 255 is NoData/outside the relevant mask.
            uncovered_target = np.full((rows, cols), 255, dtype=np.uint8)
            uncovered_target[target_positive_mask_2d] = np.where(
                count_2d[target_positive_mask_2d] > 0, 0, 1
            ).astype(np.uint8)
            uncovered_forest = np.full((rows, cols), 255, dtype=np.uint8)
            if forest_total_pixels > 0:
                uncovered_forest[forest_mask_2d] = np.where(
                    count_2d[forest_mask_2d] > 0, 0, 1
                ).astype(np.uint8)

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
            self._write_raster(
                output_uncovered_target,
                uncovered_target,
                gt,
                projection,
                gdal.GDT_Byte,
                nodata=255,
            )
            self._write_raster(
                output_uncovered_forest,
                uncovered_forest,
                gt,
                projection,
                gdal.GDT_Byte,
                nodata=255,
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

            target_positive_flat = target_positive_mask_2d.ravel()
            forest_flat = forest_mask_2d.ravel()
            target_visible_pixels = int(np.count_nonzero(target_positive_flat & (coverage_count > 0)))
            target_redundant_pixels = int(np.count_nonzero(target_positive_flat & (coverage_count >= 2)))
            target_uncovered_pixels = max(0, target_positive_pixels - target_visible_pixels)
            target_area_binary_pct = (
                100.0 * target_visible_pixels / target_positive_pixels
                if target_positive_pixels > 0 else 0.0
            )
            target_area_redundancy_pct = (
                100.0 * target_redundant_pixels / target_positive_pixels
                if target_positive_pixels > 0 else 0.0
            )
            weighted_redundancy_pct = 100.0 * float(
                np.sum(risk_flat[coverage_count >= 2], dtype=np.float64) / total_risk
            )

            forest_visible_pixels = int(np.count_nonzero(forest_flat & (coverage_count > 0)))
            forest_redundant_pixels = int(np.count_nonzero(forest_flat & (coverage_count >= 2)))
            forest_uncovered_pixels = max(0, forest_total_pixels - forest_visible_pixels)
            forest_bin_pct = (
                100.0 * forest_visible_pixels / forest_total_pixels
                if forest_total_pixels > 0 else 0.0
            )
            forest_redundancy_pct = (
                100.0 * forest_redundant_pixels / forest_total_pixels
                if forest_total_pixels > 0 else 0.0
            )
            forest_eff_pct = (
                100.0 * float(np.sum(current_quality[forest_flat], dtype=np.float64)) / forest_total_pixels
                if forest_total_pixels > 0 else 0.0
            )

            final_target_coverage_reached = int(
                use_target_coverage and final_binary + 1e-9 >= target_coverage
            )
            selected_grid_count = sum(1 for c in refined_selected if c.get("candidate_type") == "GRID")
            selected_shoulder_count = sum(1 for c in refined_selected if c.get("candidate_type") == "SHOULDER")
            selected_supplied_count = sum(1 for c in refined_selected if c.get("candidate_type") == "SUPPLIED")
            usable_grid_count = sum(1 for c in usable_candidates if c.get("candidate_type") == "GRID")
            usable_shoulder_count = sum(1 for c in usable_candidates if c.get("candidate_type") == "SHOULDER")
            usable_supplied_count = sum(1 for c in usable_candidates if c.get("candidate_type") == "SUPPLIED")

            os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
            with open(summary_path, "w", newline="", encoding="utf-8-sig") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["setting", "value"])
                writer.writerow(["candidate_count", len(cache)])
                writer.writerow(["candidate_count_before_viewshed", len(candidates)])
                writer.writerow(["candidate_generation_mode", candidate_mode_label])
                writer.writerow(["grid_spacing_m", grid_spacing])
                writer.writerow(["grid_candidates_generated", grid_candidate_count_generated])
                writer.writerow(["shoulder_candidates_generated", shoulder_candidate_count_generated])
                writer.writerow(["final_grid_candidates_before_viewshed", final_grid_candidate_count])
                writer.writerow(["final_shoulder_candidates_before_viewshed", final_shoulder_candidate_count])
                writer.writerow(["usable_grid_candidates", usable_grid_count])
                writer.writerow(["usable_shoulder_candidates", usable_shoulder_count])
                writer.writerow(["usable_supplied_candidates", usable_supplied_count])
                writer.writerow(["terrain_search_step_m", terrain_search_step])
                writer.writerow(["shoulder_probe_distance_m", shoulder_probe_distance])
                writer.writerow(["shoulder_max_site_slope_deg", shoulder_max_site_slope])
                writer.writerow(["shoulder_min_downhill_slope_deg", shoulder_min_downhill_slope])
                writer.writerow(["shoulder_max_back_slope_deg", shoulder_max_back_slope])
                writer.writerow(["shoulder_min_terrain_break_deg", shoulder_min_break])
                writer.writerow(["shoulder_min_spacing_m", shoulder_min_spacing])
                writer.writerow(["max_shoulder_candidates", max_shoulder_candidates])
                writer.writerow(["selection_mode", "target binary coverage" if use_target_coverage else "fixed camera count"])
                writer.writerow(["selection_objective", selection_objective_label])
                writer.writerow(["marginal_gain_meaning", "weighted newly covered risk plus tiny quality tie-break" if use_binary_objective else "weighted quality improvement"])
                writer.writerow(["max_camera_count", num_cameras])
                writer.writerow(["selected_camera_count", len(refined_selected)])
                writer.writerow(["selected_grid_camera_count", selected_grid_count])
                writer.writerow(["selected_shoulder_camera_count", selected_shoulder_count])
                writer.writerow(["selected_supplied_camera_count", selected_supplied_count])
                writer.writerow(["requested_binary_coverage_pct", target_coverage])
                writer.writerow(["target_coverage_reached_after_greedy", int(target_coverage_reached_initial)])
                writer.writerow(["target_coverage_reached_at_rank", target_coverage_reached_rank])
                writer.writerow(["target_coverage_preserved_after_refine", final_target_coverage_reached])
                writer.writerow(["initial_binary_after_greedy_pct", initial_binary_after_greedy])
                writer.writerow(["target_polygon_used", int(target_area_source is not None)])
                writer.writerow(["target_polygon_pixels", target_polygon_pixels])
                writer.writerow(["target_positive_risk_pixels", target_risk_pixels])
                writer.writerow(["target_total_weighted_risk", total_risk])
                writer.writerow(["coverage_denominator", "target polygon" if target_area_source is not None else "whole raster positive-risk area"])
                writer.writerow(["horizontal_fov_deg", fov])
                writer.writerow(["azimuth_step_deg", az_step])
                writer.writerow(["observer_height_m", observer_height])
                writer.writerow(["target_height_m", target_height])
                writer.writerow(["max_distance_m", max_distance])
                writer.writerow(["target_candidate_buffer_m", candidate_search_buffer])
                writer.writerow(["candidate_filter_mode", "actual buffered polygon" if candidate_allowed_mask is not None else "whole raster / supplied candidates"])
                writer.writerow(["candidate_search_limited_to_target_neighbourhood", int(candidate_allowed_mask is not None)])
                writer.writerow(["candidate_allowed_buffer_pixels", candidate_allowed_pixels])
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
                writer.writerow(["redundant_weighted_coverage_pct_count_ge_2", weighted_redundancy_pct])
                writer.writerow(["target_positive_area_pixels", target_positive_pixels])
                writer.writerow(["target_visible_area_pixels_count_ge_1", target_visible_pixels])
                writer.writerow(["target_uncovered_area_pixels_count_0", target_uncovered_pixels])
                writer.writerow(["target_redundant_area_pixels_count_ge_2", target_redundant_pixels])
                writer.writerow(["target_plain_area_bin_pct_count_ge_1", target_area_binary_pct])
                writer.writerow(["target_plain_area_redundancy_pct_count_ge_2", target_area_redundancy_pct])
                writer.writerow(["target_uncovered_area_ha", target_uncovered_pixels * pixel_area_ha])
                writer.writerow(["forest_class10_pixels_inside_denominator", forest_total_pixels])
                writer.writerow(["forest_class10_visible_pixels_count_ge_1", forest_visible_pixels])
                writer.writerow(["forest_class10_uncovered_pixels_count_0", forest_uncovered_pixels])
                writer.writerow(["forest_class10_redundant_pixels_count_ge_2", forest_redundant_pixels])
                writer.writerow(["forest_class10_bin_pct_count_ge_1", forest_bin_pct])
                writer.writerow(["forest_class10_eff_pct", forest_eff_pct])
                writer.writerow(["forest_class10_redundancy_pct_count_ge_2", forest_redundancy_pct])
                writer.writerow(["forest_class10_uncovered_area_ha", forest_uncovered_pixels * pixel_area_ha])
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
                        "initial_candidate_type",
                        "initial_shoulder_score",
                        "initial_shoulder_azimuth_deg",
                        "initial_downhill_slope_deg",
                        "initial_back_slope_deg",
                        "initial_terrain_break_deg",
                        "initial_site_slope_deg",
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
                            camera.get("candidate_type", "GRID"),
                            camera.get("shoulder_score"),
                            camera.get("shoulder_azimuth_deg"),
                            camera.get("downhill_slope_deg"),
                            camera.get("back_slope_deg"),
                            camera.get("terrain_break_deg"),
                            camera.get("site_slope_deg"),
                            camera["gain"],
                            camera["effective_pct"],
                            camera["binary_pct"],
                        ]
                    )

            feedback.pushInfo(f"CSV summary saved at: {summary_path}")
            feedback.setProgress(100.0)
            feedback.pushInfo(
                f"Ολοκληρώθηκε: {len(refined_selected)} κάμερες "
                f"(GRID={selected_grid_count}, SHOULDER={selected_shoulder_count}, SUPPLIED={selected_supplied_count}), "
                f"τελική eff_pct {final_effective:.2f}%, τελική bin_pct {final_binary:.2f}%, "
                f"forest class 10 bin_pct {forest_bin_pct:.2f}%, "
                f"τελικό score {100.0 * final_system_score / max(initial_system_score, 1e-12):.2f}% "
                f"του αρχικού coarse score."
            )
            if use_target_coverage:
                if final_binary + 1e-9 >= target_coverage:
                    feedback.pushInfo(
                        f"Ο στόχος bin_pct {target_coverage:.2f}% επιτεύχθηκε με {len(refined_selected)} κάμερες."
                    )
                else:
                    feedback.pushWarning(
                        f"Ο στόχος bin_pct {target_coverage:.2f}% δεν επιτεύχθηκε μέσα στο όριο των {num_cameras} καμερών "
                        f"(τελικό bin_pct={final_binary:.2f}%)."
                    )

            return {
                self.OUT_CANDIDATES: candidate_dest,
                self.OUT_INITIAL_CAMERAS: initial_dest,
                self.OUT_CAMERAS: camera_dest,
                self.OUT_MOVES: move_dest,
                self.OUT_FOV: fov_dest,
                self.OUT_QUALITY: output_quality,
                self.OUT_COUNT: output_count,
                self.OUT_REMAINING_RISK: output_remaining,
                self.OUT_UNCOVERED_TARGET: output_uncovered_target,
                self.OUT_UNCOVERED_FOREST: output_uncovered_forest,
                self.OUT_SUMMARY: summary_path,
            }

        finally:
            dem_ds = None
            lc_ds = None
            shutil.rmtree(temp_dir, ignore_errors=True)
