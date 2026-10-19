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
- optional target polygon, actual buffered-polygon candidate filtering, binary-vs-quality selection objective, optional target coverage stopping rule, uncovered-target rasters, and forest-only diagnostics.

The algorithm generates grid/terrain-shoulder candidates, calculates one cached 360°
viewshed per usable candidate, selects an initial greedy solution, and can then
rearrange that solution with azimuth re-optimization, pruning, 1-swap and
destroy-and-repair local search. If a target polygon is supplied, only weighted
vegetation inside that polygon contributes to the optimization and coverage
percentages, while camera candidates are generated only inside the actual buffered
target polygon.

Designed for QGIS 3.34+ with the GDAL Processing provider enabled.

Version 8.2 keeps the quiet viewshed/greedy logs and the post-greedy binary
rearrangement phase: azimuth coordinate optimisation, pruning, 1-swap and
destroy-and-repair local search. The rearrangement reuses the cached viewsheds;
it does not rerun GDAL viewshed for the original candidate pool. The legacy
uphill refinement phase has been removed; terrain-aware placement is handled by
shoulder candidates plus the rearrangement search. Version 8.2 can persist the expensive
visibility cache to a compressed .npz file and reload it in later runs, skipping candidate
generation and all GDAL viewshed calculations when the DEM/target/visibility geometry match.
"""

import csv
import hashlib
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
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterEnum,
    QgsProcessingParameterBoolean,
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
    REARRANGE_ENABLE = "REARRANGE_ENABLE"
    REARRANGE_MAX_PASSES = "REARRANGE_MAX_PASSES"
    REARRANGE_AZ_PASSES = "REARRANGE_AZ_PASSES"
    REARRANGE_SWAP_TRIALS = "REARRANGE_SWAP_TRIALS"
    REARRANGE_DESTROY_SIZE = "REARRANGE_DESTROY_SIZE"
    REARRANGE_DESTROY_ATTEMPTS = "REARRANGE_DESTROY_ATTEMPTS"
    REARRANGE_SHORTLIST = "REARRANGE_SHORTLIST"
    USE_VIS_CACHE = "USE_VIS_CACHE"
    VIS_CACHE_INPUT = "VIS_CACHE_INPUT"

    OUT_CANDIDATES = "OUT_CANDIDATES"
    OUT_INITIAL_CAMERAS = "OUT_INITIAL_CAMERAS"
    OUT_CAMERAS = "OUT_CAMERAS"
    OUT_FOV = "OUT_FOV"
    OUT_QUALITY = "OUT_QUALITY"
    OUT_COUNT = "OUT_COUNT"
    OUT_REMAINING_RISK = "OUT_REMAINING_RISK"
    OUT_UNCOVERED_TARGET = "OUT_UNCOVERED_TARGET"
    OUT_UNCOVERED_FOREST = "OUT_UNCOVERED_FOREST"
    OUT_SUMMARY = "OUT_SUMMARY"
    OUT_VIS_CACHE = "OUT_VIS_CACHE"

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
        return "lesvos_fire_camera_optimizer_hybrid_rearrange_v8_3_overlap_destroy"

    def displayName(self):
        return self.tr("Βελτιστοποίηση θέσεων καμερών – hybrid + rearrangement v8.3 (overlap-aware destroy)")

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
            "βήμα υποψήφιων θέσεων=3000 m, βήμα azimuth=30°. "
            "Η v8 μπορεί μετά το αρχικό greedy να αναδιατάξει το σύνολο των καμερών: "
            "ξαναβελτιστοποιεί azimuths, αφαιρεί περιττές κάμερες, κάνει 1-swap και "
            "destroy-and-repair ώστε να προσπαθήσει να πετύχει τον ίδιο binary στόχο με λιγότερες κάμερες. "
            "Στη v8.3 τα destroy neighborhoods είναι κυρίως overlap/spatial-aware, ώστε οι κάμερες που καταστρέφονται μαζί να αλληλεπιδρούν πραγματικά. "
            "Η rearrangement φάση επαναχρησιμοποιεί τα ήδη υπολογισμένα viewsheds. "
            "Δεν υπάρχει πλέον ξεχωριστό uphill refinement: τα terrain-aware σημεία προκύπτουν από τα shoulder candidates. "
            "Η v8.2 μπορεί να αποθηκεύσει visibility cache (.npz) μετά τα viewsheds και να το φορτώσει σε επόμενο run, "
            "ώστε να παραλείπονται πλήρως candidate generation και GDAL viewsheds όταν τα σχετικά inputs είναι συμβατά. "
            "Η v8.3 κάνει τα destroy groups overlap-aware: οι περισσότερες ομάδες χτίζονται γύρω από αδύναμες κάμερες με βάση "
            "το πραγματικό weighted overlap των τρεχόντων FOV masks και, συμπληρωματικά, τη χωρική γειτνίαση. "
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
                maxValue=500,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TARGET_COVERAGE,
                self.tr("Στόχος κάλυψης bin_pct (%) - 0 = χρήση όλων των καμερών"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
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
            QgsProcessingParameterBoolean(
                self.USE_VIS_CACHE,
                self.tr("Χρήση υπάρχοντος visibility cache (.npz) - παράλειψη candidate generation + viewsheds"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.VIS_CACHE_INPUT,
                self.tr("Υπάρχον visibility cache (.npz)"),
                behavior=QgsProcessingParameterFile.File,
                fileFilter="NumPy visibility cache (*.npz)",
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.REARRANGE_ENABLE,
                self.tr("Μετά το greedy: αναδιάταξη / local search για λιγότερες κάμερες"),
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.REARRANGE_MAX_PASSES,
                self.tr("Rearrangement: μέγιστοι κύκλοι local search"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
                minValue=0,
                maxValue=20,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.REARRANGE_AZ_PASSES,
                self.tr("Rearrangement: passes επαναβελτιστοποίησης azimuth όλων των καμερών"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=2,
                minValue=0,
                maxValue=10,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.REARRANGE_SWAP_TRIALS,
                self.tr("Rearrangement: πόσες αδύναμες κάμερες να εξετάζονται για 1-swap ανά κύκλο"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=4,
                minValue=0,
                maxValue=50,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.REARRANGE_DESTROY_SIZE,
                self.tr("Rearrangement: πόσες κάμερες αφαιρεί κάθε destroy-and-repair"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=2,
                maxValue=20,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.REARRANGE_DESTROY_ATTEMPTS,
                self.tr("Rearrangement: destroy-and-repair δοκιμές ανά κύκλο"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=6,
                minValue=0,
                maxValue=50,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.REARRANGE_SHORTLIST,
                self.tr("Rearrangement: top candidate θέσεις που ελέγχονται ακριβώς σε κάθε repair"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=300,
                minValue=25,
                maxValue=5000,
            )
        )

        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUT_VIS_CACHE,
                self.tr("Προαιρετικά: αποθήκευση visibility cache για επόμενα runs (.npz)"),
                fileFilter="NumPy visibility cache (*.npz)",
                optional=True,
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
                self.tr("Αρχική greedy λύση (πριν το rearrangement)"),
                QgsProcessing.TypeVectorPoint,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUT_CAMERAS,
                self.tr("Τελικές θέσεις καμερών (μετά το rearrangement)"),
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
        # q=1 near the camera and q≈0.368 at maximum range.
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
    def _sha256_array(array):
        arr = np.ascontiguousarray(array)
        return hashlib.sha256(memoryview(arr).cast("B")).hexdigest()

    @classmethod
    def _dem_fingerprint(cls, dem, dem_valid):
        h = hashlib.sha256()
        dem_bytes = np.ascontiguousarray(dem.astype(np.float32, copy=False))
        valid_bits = np.packbits(np.ascontiguousarray(dem_valid).ravel().astype(np.uint8, copy=False))
        h.update(memoryview(dem_bytes).cast("B"))
        h.update(memoryview(valid_bits).cast("B"))
        return h.hexdigest()

    @staticmethod
    def _risk_positive_fingerprint(risk_flat):
        positive_bits = np.packbits((np.asarray(risk_flat) > 0).astype(np.uint8, copy=False))
        return hashlib.sha256(memoryview(np.ascontiguousarray(positive_bits)).cast("B")).hexdigest()

    @staticmethod
    def _candidate_numeric(candidate, key):
        value = candidate.get(key)
        return np.nan if value is None else float(value)

    def _save_visibility_cache(
        self, path, candidates, usable_source_indices, cache, rows, cols, gt, projection,
        dem_crs_authid, observer_height, target_height, max_distance, dem_hash,
        risk_positive_hash, candidate_mode_label, generation_metadata, feedback,
    ):
        if not path:
            return ""
        path = os.path.abspath(path)
        if not path.lower().endswith(".npz"):
            path += ".npz"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        payload = {
            "cache_format_version": np.asarray([2], dtype=np.int32),
            "rows": np.asarray([rows], dtype=np.int32),
            "cols": np.asarray([cols], dtype=np.int32),
            "geotransform": np.asarray(gt, dtype=np.float64),
            "projection": np.asarray([projection], dtype=np.str_),
            "dem_crs_authid": np.asarray([dem_crs_authid], dtype=np.str_),
            "observer_height_m": np.asarray([observer_height], dtype=np.float64),
            "target_height_m": np.asarray([target_height], dtype=np.float64),
            "max_distance_m": np.asarray([max_distance], dtype=np.float64),
            "dem_sha256": np.asarray([dem_hash], dtype=np.str_),
            "risk_positive_sha256": np.asarray([risk_positive_hash], dtype=np.str_),
            "candidate_mode_label": np.asarray([candidate_mode_label], dtype=np.str_),
            "candidate_count_before_viewshed": np.asarray([len(candidates)], dtype=np.int32),
            "usable_candidate_count": np.asarray([len(cache)], dtype=np.int32),
            "candidate_x": np.asarray([float(c["x"]) for c in candidates], dtype=np.float64),
            "candidate_y": np.asarray([float(c["y"]) for c in candidates], dtype=np.float64),
            "candidate_row": np.asarray([int(c["row"]) for c in candidates], dtype=np.int32),
            "candidate_col": np.asarray([int(c["col"]) for c in candidates], dtype=np.int32),
            "candidate_type": np.asarray([c.get("candidate_type", "GRID") for c in candidates], dtype=np.str_),
            "candidate_shoulder_score": np.asarray([self._candidate_numeric(c, "shoulder_score") for c in candidates], dtype=np.float64),
            "candidate_shoulder_azimuth_deg": np.asarray([self._candidate_numeric(c, "shoulder_azimuth_deg") for c in candidates], dtype=np.float64),
            "candidate_downhill_slope_deg": np.asarray([self._candidate_numeric(c, "downhill_slope_deg") for c in candidates], dtype=np.float64),
            "candidate_back_slope_deg": np.asarray([self._candidate_numeric(c, "back_slope_deg") for c in candidates], dtype=np.float64),
            "candidate_terrain_break_deg": np.asarray([self._candidate_numeric(c, "terrain_break_deg") for c in candidates], dtype=np.float64),
            "candidate_site_slope_deg": np.asarray([self._candidate_numeric(c, "site_slope_deg") for c in candidates], dtype=np.float64),
            "usable_source_indices": np.asarray(usable_source_indices, dtype=np.int32),
        }
        for key, value in generation_metadata.items():
            if isinstance(value, str):
                payload[f"meta_{key}"] = np.asarray([value], dtype=np.str_)
            elif isinstance(value, (int, np.integer)):
                payload[f"meta_{key}"] = np.asarray([int(value)], dtype=np.int64)
            else:
                payload[f"meta_{key}"] = np.asarray([float(value)], dtype=np.float64)

        # Store each usable candidate as three separate arrays. This avoids a second
        # giant concatenated copy of the cache in RAM while writing the .npz file.
        for i, item in enumerate(cache):
            payload[f"idx_{i:05d}"] = np.asarray(item["indices"], dtype=np.int32)
            payload[f"bear_{i:05d}"] = np.asarray(item["bearings"], dtype=np.uint16)
            payload[f"qual_{i:05d}"] = np.asarray(item["quality"], dtype=np.float32)

        feedback.pushInfo(
            f"Αποθήκευση visibility cache: {len(cache)} usable candidates → {path}. "
            "Το compressed .npz μπορεί να είναι μεγάλο και να χρειαστεί λίγο χρόνο για εγγραφή."
        )
        np.savez_compressed(path, **payload)
        size_mb = os.path.getsize(path) / (1024.0 * 1024.0) if os.path.exists(path) else 0.0
        feedback.pushInfo(f"Visibility cache saved: {path} ({size_mb:.1f} MB).")
        return path

    def _load_visibility_cache(
        self, path, rows, cols, gt, projection, observer_height, target_height, max_distance,
        dem_hash, risk_positive_hash, feedback,
    ):
        if not path or not os.path.isfile(path):
            raise QgsProcessingException(
                "Έχει επιλεγεί χρήση visibility cache αλλά δεν δόθηκε έγκυρο αρχείο .npz."
            )
        feedback.pushInfo(f"Φόρτωση visibility cache: {path}")
        try:
            data = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise QgsProcessingException(f"Αποτυχία ανοίγματος visibility cache: {exc}")

        try:
            version = int(data["cache_format_version"][0])
            if version != 2:
                raise QgsProcessingException(
                    f"Μη υποστηριζόμενη έκδοση visibility cache: {version}. Αναμενόταν version 2."
                )
            if int(data["rows"][0]) != int(rows) or int(data["cols"][0]) != int(cols):
                raise QgsProcessingException("Το visibility cache ανήκει σε διαφορετικό raster grid (rows/cols).")
            if not np.allclose(np.asarray(data["geotransform"], dtype=np.float64), np.asarray(gt, dtype=np.float64), rtol=0.0, atol=1e-8):
                raise QgsProcessingException("Το visibility cache έχει διαφορετικό geotransform από το τρέχον DEM.")
            if str(data["projection"][0]) != str(projection):
                raise QgsProcessingException("Το visibility cache έχει διαφορετικό CRS/projection από το τρέχον DEM.")
            checks = [
                ("observer height", float(data["observer_height_m"][0]), float(observer_height)),
                ("target height", float(data["target_height_m"][0]), float(target_height)),
                ("max distance", float(data["max_distance_m"][0]), float(max_distance)),
            ]
            for label, cached_value, current_value in checks:
                if not math.isclose(cached_value, current_value, rel_tol=0.0, abs_tol=1e-9):
                    raise QgsProcessingException(
                        f"Το visibility cache δεν είναι συμβατό: {label} cache={cached_value}, current={current_value}."
                    )
            if str(data["dem_sha256"][0]) != str(dem_hash):
                raise QgsProcessingException(
                    "Το DEM έχει αλλάξει σε σχέση με το visibility cache. Χρειάζεται νέο viewshed cache."
                )
            if str(data["risk_positive_sha256"][0]) != str(risk_positive_hash):
                raise QgsProcessingException(
                    "Η positive-risk/target mask έχει αλλάξει σε σχέση με το visibility cache. "
                    "Χρειάζεται νέο cache (π.χ. άλλαξε target polygon ή classes με θετικό βάρος)."
                )

            xs = np.asarray(data["candidate_x"], dtype=np.float64)
            ys = np.asarray(data["candidate_y"], dtype=np.float64)
            rs = np.asarray(data["candidate_row"], dtype=np.int32)
            cs = np.asarray(data["candidate_col"], dtype=np.int32)
            types = np.asarray(data["candidate_type"]).astype(str)
            field_map = {
                "shoulder_score": np.asarray(data["candidate_shoulder_score"], dtype=np.float64),
                "shoulder_azimuth_deg": np.asarray(data["candidate_shoulder_azimuth_deg"], dtype=np.float64),
                "downhill_slope_deg": np.asarray(data["candidate_downhill_slope_deg"], dtype=np.float64),
                "back_slope_deg": np.asarray(data["candidate_back_slope_deg"], dtype=np.float64),
                "terrain_break_deg": np.asarray(data["candidate_terrain_break_deg"], dtype=np.float64),
                "site_slope_deg": np.asarray(data["candidate_site_slope_deg"], dtype=np.float64),
            }
            candidates = []
            for i in range(xs.size):
                candidate = {
                    "x": float(xs[i]), "y": float(ys[i]), "row": int(rs[i]), "col": int(cs[i]),
                    "candidate_type": str(types[i]),
                }
                for key, arr in field_map.items():
                    value = float(arr[i])
                    candidate[key] = None if np.isnan(value) else value
                candidates.append(candidate)

            usable_source_indices = np.asarray(data["usable_source_indices"], dtype=np.int32).tolist()
            usable_count = int(data["usable_candidate_count"][0])
            if usable_count != len(usable_source_indices):
                raise QgsProcessingException("Κατεστραμμένο visibility cache: inconsistent usable candidate count.")
            cache = []
            for i in range(usable_count):
                cache.append({
                    "indices": np.asarray(data[f"idx_{i:05d}"], dtype=np.int32),
                    "bearings": np.asarray(data[f"bear_{i:05d}"], dtype=np.uint16),
                    "quality": np.asarray(data[f"qual_{i:05d}"], dtype=np.float32),
                })
            if any(i < 0 or i >= len(candidates) for i in usable_source_indices):
                raise QgsProcessingException("Κατεστραμμένο visibility cache: invalid candidate mapping.")
            usable_candidates = [candidates[i] for i in usable_source_indices]

            metadata = {}
            for key in data.files:
                if not key.startswith("meta_"):
                    continue
                value = data[key][0]
                if isinstance(value, np.generic):
                    value = value.item()
                metadata[key[5:]] = value
            candidate_mode_label = str(data["candidate_mode_label"][0])
        finally:
            data.close()

        feedback.pushInfo(
            f"Visibility cache loaded: {len(cache)} usable / {len(candidates)} total candidates. "
            "Candidate generation και GDAL viewsheds παραλείπονται."
        )
        return candidates, usable_candidates, usable_source_indices, cache, candidate_mode_label, metadata

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

    def _orientation_binary_score(
        self,
        item,
        baseline_count,
        risk_flat,
        azimuths,
        fov,
        half_fov,
    ):
        """Best binary azimuth for one cached candidate against a fixed baseline.

        This v8 implementation is intentionally much faster than testing a large
        pixel mask once per azimuth. Bearings in the cache are integer degrees, so
        uncovered weighted risk is first accumulated into a 360-bin histogram.
        Every allowed FOV direction is then scored from that small histogram. Only
        the winning azimuth is expanded back to the full pixel mask.
        """
        indices = item["indices"]
        bearings = item["bearings"]
        quality = item["quality"]
        new_pixels = baseline_count[indices] == 0

        azimuth_array = np.asarray(azimuths, dtype=np.float32)
        if azimuth_array.size == 0:
            return None

        if np.any(new_pixels):
            new_bearings = (
                bearings[new_pixels].astype(np.int16, copy=False) % 360
            )
            new_indices = indices[new_pixels]
            new_quality = quality[new_pixels]
            weight_hist = np.bincount(
                new_bearings,
                weights=risk_flat[new_indices].astype(np.float64, copy=False),
                minlength=360,
            ).astype(np.float64, copy=False)
            quality_hist = np.bincount(
                new_bearings,
                weights=(risk_flat[new_indices] * new_quality).astype(np.float64, copy=False),
                minlength=360,
            ).astype(np.float64, copy=False)
        else:
            weight_hist = np.zeros(360, dtype=np.float64)
            quality_hist = np.zeros(360, dtype=np.float64)

        # Cache the tiny azimuth x 360-degree sector matrix. It is reused by every
        # candidate in greedy/rearrangement for the same FOV/azimuth step.
        first_az = float(azimuth_array[0]) if azimuth_array.size else 0.0
        az_delta = (
            float(azimuth_array[1] - azimuth_array[0])
            if azimuth_array.size > 1
            else 0.0
        )
        sector_key = (
            int(azimuth_array.size),
            round(first_az, 6),
            round(az_delta, 6),
            round(float(fov), 6),
        )
        sector_cache = getattr(self, "_binary_sector_matrix_cache", None)
        if sector_cache is None:
            sector_cache = {}
            self._binary_sector_matrix_cache = sector_cache
        sector_matrix = sector_cache.get(sector_key)
        if sector_matrix is None:
            degree_bins = np.arange(360, dtype=np.float32)
            if fov >= 359.999:
                sector_matrix = np.ones(
                    (azimuth_array.size, 360), dtype=np.float32
                )
            else:
                differences = np.abs(
                    (degree_bins[None, :] - azimuth_array[:, None] + 180.0) % 360.0
                    - 180.0
                )
                sector_matrix = (differences <= half_fov).astype(np.float32)
            sector_cache[sector_key] = sector_matrix

        gains = sector_matrix @ weight_hist
        quality_ties = sector_matrix @ quality_hist
        best_gain = float(np.max(gains))
        gain_close = np.isclose(gains, best_gain, rtol=0.0, atol=1e-8)
        tied_positions = np.flatnonzero(gain_close)
        if tied_positions.size == 0:
            best_position = int(np.argmax(gains))
        elif tied_positions.size == 1:
            best_position = int(tied_positions[0])
        else:
            best_position = int(
                tied_positions[np.argmax(quality_ties[tied_positions])]
            )

        best_azimuth = float(azimuth_array[best_position])
        if fov >= 359.999:
            mask = np.ones(indices.shape, dtype=bool)
        else:
            mask = self._angular_difference(bearings, best_azimuth) <= half_fov
        if not np.any(mask):
            return None

        return {
            "azimuth": best_azimuth,
            "indices": indices[mask].copy(),
            "quality": quality[mask].copy(),
            "binary_gain": best_gain,
            "quality_tiebreak": float(quality_ties[best_position]),
        }

    @staticmethod
    def _covered_weight(count, risk_flat):
        return float(np.sum(risk_flat[count > 0], dtype=np.float64))

    def _make_camera_from_candidate(
        self,
        candidate_index,
        orientation,
        usable_candidates,
        dem,
        greedy_rank_by_candidate=None,
        rearranged=1,
    ):
        candidate = usable_candidates[candidate_index]
        greedy_rank = 0
        if greedy_rank_by_candidate is not None:
            greedy_rank = int(greedy_rank_by_candidate.get(candidate_index, 0))
        return {
            "rank": 0,
            "candidate_index": int(candidate_index),
            "x": float(candidate["x"]),
            "y": float(candidate["y"]),
            "row": int(candidate["row"]),
            "col": int(candidate["col"]),
            "elev_m": float(dem[candidate["row"], candidate["col"]]),
            "azimuth": float(orientation["azimuth"]),
            "gain": float(orientation.get("binary_gain", 0.0)),
            "effective_pct": 0.0,
            "binary_pct": 0.0,
            "candidate_type": candidate.get("candidate_type", "GRID"),
            "shoulder_score": candidate.get("shoulder_score"),
            "shoulder_azimuth_deg": candidate.get("shoulder_azimuth_deg"),
            "downhill_slope_deg": candidate.get("downhill_slope_deg"),
            "back_slope_deg": candidate.get("back_slope_deg"),
            "terrain_break_deg": candidate.get("terrain_break_deg"),
            "site_slope_deg": candidate.get("site_slope_deg"),
            "indices": orientation["indices"].copy(),
            "quality": orientation["quality"].copy(),
            "greedy_rank": greedy_rank,
            "rearranged": int(rearranged),
        }

    def _candidate_shortlist_binary(
        self,
        cache,
        baseline_count,
        risk_flat,
        excluded_candidate_indices,
        shortlist_size,
    ):
        """Rank candidates by a cheap 360-degree upper bound before azimuth search."""
        scored = []
        excluded = set(excluded_candidate_indices)
        for candidate_index, item in enumerate(cache):
            if candidate_index in excluded:
                continue
            indices = item["indices"]
            new_pixels = baseline_count[indices] == 0
            if not np.any(new_pixels):
                continue
            upper_gain = float(
                np.sum(risk_flat[indices[new_pixels]], dtype=np.float64)
            )
            if upper_gain > 0.0:
                scored.append((upper_gain, candidate_index))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        if shortlist_size > 0 and len(scored) > shortlist_size:
            scored = scored[:shortlist_size]
        return [candidate_index for _, candidate_index in scored]

    def _best_repair_candidate_binary(
        self,
        cache,
        usable_candidates,
        dem,
        baseline_count,
        risk_flat,
        azimuths,
        fov,
        half_fov,
        excluded_candidate_indices,
        shortlist_size,
        greedy_rank_by_candidate,
        candidate_pool=None,
    ):
        if candidate_pool is None:
            shortlist = self._candidate_shortlist_binary(
                cache,
                baseline_count,
                risk_flat,
                excluded_candidate_indices,
                shortlist_size,
            )
        else:
            excluded = set(excluded_candidate_indices)
            shortlist = [
                int(candidate_index)
                for candidate_index in candidate_pool
                if int(candidate_index) not in excluded
            ]
        best_camera = None
        best_gain = -1.0
        best_quality_tie = -1.0

        for candidate_index in shortlist:
            orientation = self._orientation_binary_score(
                cache[candidate_index],
                baseline_count,
                risk_flat,
                azimuths,
                fov,
                half_fov,
            )
            if orientation is None:
                continue
            gain = float(orientation["binary_gain"])
            quality_tie = float(orientation["quality_tiebreak"])
            if gain <= 0.0:
                continue
            if (
                gain > best_gain + 1e-9
                or (
                    abs(gain - best_gain) <= 1e-9
                    and quality_tie > best_quality_tie + 1e-9
                )
            ):
                best_gain = gain
                best_quality_tie = quality_tie
                best_camera = self._make_camera_from_candidate(
                    candidate_index,
                    orientation,
                    usable_candidates,
                    dem,
                    greedy_rank_by_candidate=greedy_rank_by_candidate,
                    rearranged=1,
                )
                best_camera["repair_binary_gain"] = gain
                best_camera["repair_quality_tiebreak"] = quality_tie

        return best_camera

    def _azimuth_coordinate_optimize_binary(
        self,
        cameras,
        cache,
        risk_flat,
        total_risk,
        azimuths,
        fov,
        half_fov,
        max_passes,
        feedback,
        log_prefix="Azimuth re-opt",
    ):
        """Coordinate-ascent on azimuths while keeping camera positions fixed."""
        if max_passes <= 0 or not cameras:
            return cameras, 0

        _, global_count = self._compose_coverage(cameras, risk_flat.size)
        total_changes = 0

        for pass_index in range(max_passes):
            if feedback.isCanceled():
                break
            pass_changes = 0

            for camera in cameras:
                if feedback.isCanceled():
                    break
                candidate_index = int(camera["candidate_index"])
                old_indices = camera["indices"]
                old_quality = camera["quality"]
                old_azimuth = float(camera["azimuth"])

                # Remove only this camera from the baseline.
                global_count[old_indices] -= 1
                old_unique = global_count[old_indices] == 0
                old_binary_gain = float(
                    np.sum(risk_flat[old_indices[old_unique]], dtype=np.float64)
                ) if np.any(old_unique) else 0.0
                old_quality_tie = float(
                    np.sum(
                        risk_flat[old_indices[old_unique]] * old_quality[old_unique],
                        dtype=np.float64,
                    )
                ) if np.any(old_unique) else 0.0

                best = self._orientation_binary_score(
                    cache[candidate_index],
                    global_count,
                    risk_flat,
                    azimuths,
                    fov,
                    half_fov,
                )

                accept = False
                if best is not None:
                    if best["binary_gain"] > old_binary_gain + 1e-9:
                        accept = True
                    elif (
                        abs(best["binary_gain"] - old_binary_gain) <= 1e-9
                        and best["quality_tiebreak"] > old_quality_tie + 1e-6
                    ):
                        accept = True

                if accept:
                    camera["azimuth"] = float(best["azimuth"])
                    camera["indices"] = best["indices"].copy()
                    camera["quality"] = best["quality"].copy()
                    if abs(float(best["azimuth"]) - old_azimuth) > 1e-9:
                        camera["rearranged"] = 1
                        pass_changes += 1
                        total_changes += 1
                    global_count[camera["indices"]] += 1
                else:
                    global_count[old_indices] += 1

            binary_pct = 100.0 * self._covered_weight(global_count, risk_flat) / total_risk
            feedback.pushInfo(
                f"{log_prefix} pass {pass_index + 1}: {pass_changes} azimuth changes, "
                f"bin_pct={binary_pct:.4f}%."
            )
            if pass_changes == 0:
                break

        return cameras, total_changes

    def _prune_binary_target(
        self,
        cameras,
        risk_flat,
        total_risk,
        target_coverage,
        feedback,
    ):
        """Remove cameras that are unnecessary while preserving the binary target."""
        if target_coverage <= 0.0 or len(cameras) <= 1:
            return cameras, 0

        cameras = list(cameras)
        _, count = self._compose_coverage(cameras, risk_flat.size)
        covered_weight = self._covered_weight(count, risk_flat)
        target_weight = total_risk * float(target_coverage) / 100.0
        removed_count = 0

        while len(cameras) > 1 and covered_weight + 1e-9 >= target_weight:
            best_pos = None
            best_unique_weight = None

            for pos, camera in enumerate(cameras):
                indices = camera["indices"]
                unique_mask = count[indices] == 1
                unique_weight = float(
                    np.sum(risk_flat[indices[unique_mask]], dtype=np.float64)
                ) if np.any(unique_mask) else 0.0
                if covered_weight - unique_weight + 1e-9 < target_weight:
                    continue
                if best_unique_weight is None or unique_weight < best_unique_weight:
                    best_unique_weight = unique_weight
                    best_pos = pos

            if best_pos is None:
                break

            removed = cameras.pop(best_pos)
            count[removed["indices"]] -= 1
            covered_weight -= float(best_unique_weight)
            removed_count += 1
            feedback.pushInfo(
                f"Pruning: αφαιρέθηκε camera candidate {removed['candidate_index']} — "
                f"μένουν {len(cameras)}, bin_pct={100.0 * covered_weight / total_risk:.4f}%."
            )

        return cameras, removed_count

    def _one_swap_binary(
        self,
        cameras,
        cache,
        usable_candidates,
        dem,
        risk_flat,
        total_risk,
        azimuths,
        fov,
        half_fov,
        weak_trials,
        shortlist_size,
        greedy_rank_by_candidate,
        feedback,
    ):
        """Try replacing one weak camera with one better candidate+azimuth."""
        if weak_trials <= 0 or len(cameras) <= 1:
            return cameras, False, 0.0

        _, global_count = self._compose_coverage(cameras, risk_flat.size)
        current_weight = self._covered_weight(global_count, risk_flat)
        current_pct = 100.0 * current_weight / total_risk

        unique_weights = []
        for pos, camera in enumerate(cameras):
            indices = camera["indices"]
            unique_mask = global_count[indices] == 1
            unique_weight = float(
                np.sum(risk_flat[indices[unique_mask]], dtype=np.float64)
            ) if np.any(unique_mask) else 0.0
            unique_weights.append((unique_weight, pos))
        unique_weights.sort(key=lambda pair: (pair[0], pair[1]))

        best_swap = None
        best_result_pct = current_pct
        for _, pos in unique_weights[: min(weak_trials, len(unique_weights))]:
            if feedback.isCanceled():
                break
            old_camera = cameras[pos]
            baseline_count = global_count.copy()
            baseline_count[old_camera["indices"]] -= 1
            baseline_weight = self._covered_weight(baseline_count, risk_flat)
            excluded = {
                int(camera["candidate_index"])
                for idx, camera in enumerate(cameras)
                if idx != pos
            }

            replacement = self._best_repair_candidate_binary(
                cache,
                usable_candidates,
                dem,
                baseline_count,
                risk_flat,
                azimuths,
                fov,
                half_fov,
                excluded,
                shortlist_size,
                greedy_rank_by_candidate,
            )
            if replacement is None:
                continue

            result_weight = baseline_weight + float(replacement["repair_binary_gain"])
            result_pct = 100.0 * result_weight / total_risk
            if result_pct > best_result_pct + 1e-7:
                best_result_pct = result_pct
                best_swap = (pos, replacement, old_camera)

        if best_swap is None:
            return cameras, False, current_pct

        pos, replacement, old_camera = best_swap
        new_cameras = list(cameras)
        new_cameras[pos] = replacement
        feedback.pushInfo(
            f"1-swap: candidate {old_camera['candidate_index']} → {replacement['candidate_index']}, "
            f"bin_pct {current_pct:.4f}% → {best_result_pct:.4f}%."
        )
        return new_cameras, True, best_result_pct

    def _destroy_repair_binary(
        self,
        cameras,
        remove_positions,
        cache,
        usable_candidates,
        dem,
        risk_flat,
        total_risk,
        target_coverage,
        azimuths,
        fov,
        half_fov,
        shortlist_size,
        greedy_rank_by_candidate,
        feedback,
    ):
        """Destroy k cameras and greedily repair with at most k candidates.

        If the target is reached before all k slots are restored, the resulting
        solution uses fewer cameras and is returned immediately.
        """
        original_n = len(cameras)
        remove_set = set(int(pos) for pos in remove_positions)
        working = [camera for pos, camera in enumerate(cameras) if pos not in remove_set]
        _, count = self._compose_coverage(working, risk_flat.size)
        used_candidates = {int(camera["candidate_index"]) for camera in working}
        current_pct = 100.0 * self._covered_weight(count, risk_flat) / total_risk
        # Rank the repair pool once for the destroyed baseline. Reusing this pool
        # across the k repair slots avoids rescanning every cached candidate k times.
        repair_pool = self._candidate_shortlist_binary(
            cache,
            count,
            risk_flat,
            used_candidates,
            shortlist_size,
        )

        for repair_slot in range(len(remove_set)):
            if feedback.isCanceled():
                break
            replacement = self._best_repair_candidate_binary(
                cache,
                usable_candidates,
                dem,
                count,
                risk_flat,
                azimuths,
                fov,
                half_fov,
                used_candidates,
                shortlist_size,
                greedy_rank_by_candidate,
                candidate_pool=repair_pool,
            )
            if replacement is None:
                break
            working.append(replacement)
            used_candidates.add(int(replacement["candidate_index"]))
            count[replacement["indices"]] += 1
            current_pct = 100.0 * self._covered_weight(count, risk_flat) / total_risk

            if (
                target_coverage > 0.0
                and current_pct + 1e-9 >= target_coverage
                and len(working) < original_n
            ):
                return working, current_pct, True

        return (
            working,
            current_pct,
            target_coverage > 0.0
            and len(working) < original_n
            and current_pct + 1e-9 >= target_coverage,
        )

    def _destroy_groups(self, cameras, risk_flat, destroy_size, attempts, pass_index):
        """Build coherent destroy neighborhoods for large-neighborhood search.

        v8.3 deliberately avoids making most destroy groups from unrelated cameras.
        Groups are generated mainly from:
          1) overlap neighborhoods around weak cameras,
          2) spatial neighborhoods around weak cameras,
          3) overlap neighborhoods around deterministic random seeds,
        with the old weakest/tail/weak-window groups retained as fallback diversity.

        Pairwise overlap is weighted by the risk raster and normalized by the
        smaller camera footprint:

            overlap(i,j) = W(C_i ∩ C_j) / min(W(C_i), W(C_j))

        This score is 1 when the smaller camera's weighted footprint is completely
        contained in the other camera and 0 when they do not overlap at all.
        """
        if attempts <= 0 or len(cameras) <= 2:
            return []
        k = min(int(destroy_size), len(cameras) - 1)
        if k <= 0:
            return []

        n = len(cameras)
        _, count = self._compose_coverage(cameras, risk_flat.size)

        # Weakness = weighted risk that would become uncovered if this camera alone
        # were removed from the current solution.
        weak = []
        unique_weight_by_pos = np.zeros(n, dtype=np.float64)
        total_weight_by_pos = np.zeros(n, dtype=np.float64)
        xs = np.zeros(n, dtype=np.float64)
        ys = np.zeros(n, dtype=np.float64)

        for pos, camera in enumerate(cameras):
            indices = camera["indices"]
            unique_mask = count[indices] == 1
            unique_weight = float(
                np.sum(risk_flat[indices[unique_mask]], dtype=np.float64)
            ) if np.any(unique_mask) else 0.0
            total_weight = float(np.sum(risk_flat[indices], dtype=np.float64))
            unique_weight_by_pos[pos] = unique_weight
            total_weight_by_pos[pos] = total_weight
            xs[pos] = float(camera["x"])
            ys[pos] = float(camera["y"])
            weak.append((unique_weight, pos))

        weak_order = [pos for _, pos in sorted(weak, key=lambda pair: (pair[0], pair[1]))]

        # Overlap is computed lazily and cached within this rearrangement pass.
        # Camera indices are unique raster-cell ids, so assume_unique=True avoids
        # unnecessary sorting inside np.intersect1d.
        pair_overlap_cache = {}

        def pair_overlap(i, j):
            i = int(i)
            j = int(j)
            if i == j:
                return 1.0
            key = (i, j) if i < j else (j, i)
            cached = pair_overlap_cache.get(key)
            if cached is not None:
                return cached
            denom = min(total_weight_by_pos[i], total_weight_by_pos[j])
            if denom <= 0.0:
                pair_overlap_cache[key] = 0.0
                return 0.0
            common = np.intersect1d(
                cameras[i]["indices"],
                cameras[j]["indices"],
                assume_unique=True,
            )
            if common.size == 0:
                score = 0.0
            else:
                common_weight = float(np.sum(risk_flat[common], dtype=np.float64))
                score = common_weight / denom
            score = float(max(0.0, min(1.0, score)))
            pair_overlap_cache[key] = score
            return score

        def min_distance_sq_to_group(pos, group):
            dx = xs[pos] - xs[np.asarray(group, dtype=np.int64)]
            dy = ys[pos] - ys[np.asarray(group, dtype=np.int64)]
            return float(np.min(dx * dx + dy * dy))

        def overlap_group(seed):
            """Grow a connected group by strongest overlap; use distance as fallback."""
            group = [int(seed)]
            connection_scores = []
            while len(group) < k:
                best_pos = None
                best_key = None
                best_connection = 0.0
                for pos in range(n):
                    if pos in group:
                        continue
                    overlaps = [pair_overlap(pos, member) for member in group]
                    max_overlap = max(overlaps) if overlaps else 0.0
                    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
                    d2 = min_distance_sq_to_group(pos, group)
                    # Primary: actual FOV overlap. Secondary: coherence with the
                    # whole group. Tertiary: proximity and low unique contribution.
                    key = (
                        max_overlap,
                        mean_overlap,
                        -d2,
                        -unique_weight_by_pos[pos],
                        -pos,
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_pos = pos
                        best_connection = max_overlap
                if best_pos is None:
                    break
                group.append(int(best_pos))
                connection_scores.append(float(best_connection))
            mean_connection = float(np.mean(connection_scores)) if connection_scores else 0.0
            return group, mean_connection

        def spatial_group(seed):
            """Seed plus the nearest k-1 current cameras in map coordinates."""
            seed = int(seed)
            dx = xs - xs[seed]
            dy = ys - ys[seed]
            d2 = dx * dx + dy * dy
            order = np.argsort(d2, kind="stable")
            group = [int(pos) for pos in order[:k]]
            max_distance = float(np.sqrt(np.max(d2[np.asarray(group, dtype=np.int64)]))) if group else 0.0
            return group, max_distance

        groups = []
        seen = set()

        def add_group(values, kind):
            group = tuple(sorted(set(int(v) for v in values)))
            if len(group) != k or group in seen or len(groups) >= attempts:
                return False
            seen.add(group)
            groups.append((group, str(kind)))
            return True

        # Rough mix for 20 attempts:
        #   ~10 weak-seeded overlap groups
        #   ~4 weak-seeded spatial groups
        #   ~3 random-seeded overlap groups
        #   remaining slots = classic weak/tail/random-spatial fallbacks
        overlap_target = min(attempts, max(1, int(np.ceil(attempts * 0.50))))
        spatial_target = min(attempts, overlap_target + max(1, int(np.ceil(attempts * 0.20))))
        random_overlap_target = min(attempts, spatial_target + max(1, int(np.ceil(attempts * 0.15))))

        # 1) Weak-seeded overlap neighborhoods. These are the most targeted groups.
        seed_cursor = 0
        while len(groups) < overlap_target and seed_cursor < len(weak_order):
            seed = weak_order[seed_cursor]
            seed_cursor += 1
            group, mean_connection = overlap_group(seed)
            add_group(
                group,
                f"overlap_weak(seed_cand={cameras[seed]['candidate_index']},mean_link={mean_connection:.3f})",
            )

        # 2) Weak-seeded spatial neighborhoods. Useful where current FOVs barely
        # overlap but nearby cameras still compete for the same local terrain.
        seed_cursor = 0
        while len(groups) < spatial_target and seed_cursor < len(weak_order):
            seed = weak_order[seed_cursor]
            seed_cursor += 1
            group, radius_m = spatial_group(seed)
            add_group(
                group,
                f"spatial_weak(seed_cand={cameras[seed]['candidate_index']},radius={radius_m:.0f}m)",
            )

        # 3) Deterministic random seeds, but the group around each seed is still
        # overlap-coherent rather than five unrelated random cameras.
        rng = np.random.default_rng(20260919 + int(pass_index))
        random_seed_order = rng.permutation(n)
        seed_cursor = 0
        while len(groups) < random_overlap_target and seed_cursor < len(random_seed_order):
            seed = int(random_seed_order[seed_cursor])
            seed_cursor += 1
            group, mean_connection = overlap_group(seed)
            add_group(
                group,
                f"overlap_random_seed(seed_cand={cameras[seed]['candidate_index']},mean_link={mean_connection:.3f})",
            )

        # 4) Keep a little of the old strategy for diversity / escape from a bad
        # local neighborhood. These only fill remaining attempt slots.
        add_group(weak_order[:k], "classic_weakest")
        if len(cameras) >= k:
            add_group(range(len(cameras) - k, len(cameras)), "classic_tail")

        offset = 1
        while len(groups) < attempts and offset + k <= len(weak_order):
            add_group(weak_order[offset : offset + k], f"classic_weak_window(offset={offset})")
            offset += max(1, k // 2)

        # Final fallback: random seed + spatial neighbors, still geographically coherent.
        tries = 0
        while len(groups) < attempts and tries < attempts * 20:
            tries += 1
            seed = int(rng.integers(0, n))
            group, radius_m = spatial_group(seed)
            add_group(
                group,
                f"spatial_random_seed(seed_cand={cameras[seed]['candidate_index']},radius={radius_m:.0f}m)",
            )

        return groups[:attempts]

    def _local_rearrange_binary(
        self,
        selected,
        cache,
        usable_candidates,
        dem,
        risk_flat,
        total_risk,
        target_coverage,
        azimuths,
        fov,
        half_fov,
        max_passes,
        azimuth_passes,
        swap_trials,
        destroy_size,
        destroy_attempts,
        shortlist_size,
        feedback,
    ):
        """Post-greedy local / large-neighborhood search for the binary objective."""
        cameras = []
        for camera in selected:
            copied = dict(camera)
            copied["indices"] = camera["indices"].copy()
            copied["quality"] = camera["quality"].copy()
            copied["greedy_rank"] = int(camera.get("rank", 0))
            copied["rearranged"] = 0
            cameras.append(copied)

        greedy_rank_by_candidate = {
            int(camera["candidate_index"]): int(camera.get("rank", 0))
            for camera in selected
        }
        stats = {
            "azimuth_changes": 0,
            "one_swaps": 0,
            "destroy_repairs": 0,
            "cameras_removed": 0,
            "passes_completed": 0,
        }

        _, start_count = self._compose_coverage(cameras, risk_flat.size)
        start_pct = 100.0 * self._covered_weight(start_count, risk_flat) / total_risk
        feedback.pushInfo(
            f"Rearrangement start: {len(cameras)} cameras, bin_pct={start_pct:.4f}%. "
            f"No new GDAL viewsheds will be calculated."
        )

        cameras, changes = self._azimuth_coordinate_optimize_binary(
            cameras,
            cache,
            risk_flat,
            total_risk,
            azimuths,
            fov,
            half_fov,
            azimuth_passes,
            feedback,
            log_prefix="Initial azimuth re-opt",
        )
        stats["azimuth_changes"] += changes

        before_prune = len(cameras)
        cameras, pruned = self._prune_binary_target(
            cameras, risk_flat, total_risk, target_coverage, feedback
        )
        stats["cameras_removed"] += pruned

        for pass_index in range(max(0, int(max_passes))):
            if feedback.isCanceled():
                break
            stats["passes_completed"] += 1
            changed_this_pass = len(cameras) < before_prune
            _, pass_count = self._compose_coverage(cameras, risk_flat.size)
            pass_pct = 100.0 * self._covered_weight(pass_count, risk_flat) / total_risk
            feedback.pushInfo(
                f"Rearrangement pass {pass_index + 1}/{max_passes}: "
                f"{len(cameras)} cameras, bin_pct={pass_pct:.4f}%."
            )
            feedback.setProgress(
                90.0 + 8.0 * (pass_index + 1) / max(1, int(max_passes))
            )

            # 1-swap improves the arrangement at the same camera count.
            swapped_cameras, swapped, _ = self._one_swap_binary(
                cameras,
                cache,
                usable_candidates,
                dem,
                risk_flat,
                total_risk,
                azimuths,
                fov,
                half_fov,
                swap_trials,
                shortlist_size,
                greedy_rank_by_candidate,
                feedback,
            )
            if swapped:
                cameras = swapped_cameras
                stats["one_swaps"] += 1
                changed_this_pass = True
                cameras, changes = self._azimuth_coordinate_optimize_binary(
                    cameras,
                    cache,
                    risk_flat,
                    total_risk,
                    azimuths,
                    fov,
                    half_fov,
                    1,
                    feedback,
                    log_prefix="Post-swap azimuth re-opt",
                )
                stats["azimuth_changes"] += changes
                old_n = len(cameras)
                cameras, pruned = self._prune_binary_target(
                    cameras, risk_flat, total_risk, target_coverage, feedback
                )
                if pruned:
                    stats["cameras_removed"] += pruned
                    changed_this_pass = True
                    feedback.pushInfo(
                        f"Post-swap pruning: {old_n} → {len(cameras)} cameras."
                    )

            # Large-neighborhood destroy-and-repair. Prefer any solution that reaches
            # the target with fewer cameras; otherwise keep the best same-count repair.
            groups = self._destroy_groups(
                cameras, risk_flat, destroy_size, destroy_attempts, pass_index
            )
            _, base_count = self._compose_coverage(cameras, risk_flat.size)
            base_pct = 100.0 * self._covered_weight(base_count, risk_flat) / total_risk
            best_same = None
            best_same_pct = base_pct
            accepted_reduction = False

            for attempt_index, group_info in enumerate(groups, start=1):
                if feedback.isCanceled():
                    break
                group, group_kind = group_info
                feedback.pushInfo(
                    f"Destroy/repair {attempt_index}/{len(groups)} [{group_kind}]: "
                    f"remove {len(group)} cameras from a {len(cameras)}-camera solution."
                )
                trial, trial_pct, reduced = self._destroy_repair_binary(
                    cameras,
                    group,
                    cache,
                    usable_candidates,
                    dem,
                    risk_flat,
                    total_risk,
                    target_coverage,
                    azimuths,
                    fov,
                    half_fov,
                    shortlist_size,
                    greedy_rank_by_candidate,
                    feedback,
                )
                if reduced and len(trial) < len(cameras):
                    old_n = len(cameras)
                    cameras = trial
                    stats["destroy_repairs"] += 1
                    stats["cameras_removed"] += old_n - len(cameras)
                    changed_this_pass = True
                    accepted_reduction = True
                    feedback.pushInfo(
                        f"Destroy/repair SUCCESS: {old_n} → {len(cameras)} cameras, "
                        f"bin_pct={trial_pct:.4f}%."
                    )
                    break
                if len(trial) == len(cameras) and trial_pct > best_same_pct + 1e-7:
                    best_same = trial
                    best_same_pct = trial_pct

            if not accepted_reduction and best_same is not None:
                cameras = best_same
                stats["destroy_repairs"] += 1
                changed_this_pass = True
                feedback.pushInfo(
                    f"Destroy/repair improved same-count solution: bin_pct "
                    f"{base_pct:.4f}% → {best_same_pct:.4f}%."
                )

            if accepted_reduction or best_same is not None:
                cameras, changes = self._azimuth_coordinate_optimize_binary(
                    cameras,
                    cache,
                    risk_flat,
                    total_risk,
                    azimuths,
                    fov,
                    half_fov,
                    1,
                    feedback,
                    log_prefix="Post-repair azimuth re-opt",
                )
                stats["azimuth_changes"] += changes
                old_n = len(cameras)
                cameras, pruned = self._prune_binary_target(
                    cameras, risk_flat, total_risk, target_coverage, feedback
                )
                if pruned:
                    stats["cameras_removed"] += pruned
                    changed_this_pass = True
                    feedback.pushInfo(
                        f"Post-repair pruning: {old_n} → {len(cameras)} cameras."
                    )

            if not changed_this_pass:
                feedback.pushInfo("Rearrangement: no further improvement in this pass; stopping local search.")
                break

            before_prune = len(cameras)

        # One final azimuth pass and prune after all accepted set changes.
        cameras, changes = self._azimuth_coordinate_optimize_binary(
            cameras,
            cache,
            risk_flat,
            total_risk,
            azimuths,
            fov,
            half_fov,
            1 if azimuth_passes > 0 else 0,
            feedback,
            log_prefix="Final azimuth re-opt",
        )
        stats["azimuth_changes"] += changes
        cameras, pruned = self._prune_binary_target(
            cameras, risk_flat, total_risk, target_coverage, feedback
        )
        stats["cameras_removed"] += pruned

        # Final ranks are ranks of the rearranged solution. greedy_rank is retained
        # separately so the CSV can show where a camera came from.
        for rank, camera in enumerate(cameras, start=1):
            camera["rank"] = rank

        _, final_count = self._compose_coverage(cameras, risk_flat.size)
        final_pct = 100.0 * self._covered_weight(final_count, risk_flat) / total_risk
        stats["final_camera_count"] = len(cameras)
        stats["final_binary_pct"] = final_pct
        feedback.pushInfo(
            f"Rearrangement finished: {len(selected)} → {len(cameras)} cameras, "
            f"bin_pct {start_pct:.4f}% → {final_pct:.4f}%."
        )
        return cameras, stats

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
        rearrange_enable = self.parameterAsBool(parameters, self.REARRANGE_ENABLE, context)
        rearrange_max_passes = self.parameterAsInt(parameters, self.REARRANGE_MAX_PASSES, context)
        rearrange_az_passes = self.parameterAsInt(parameters, self.REARRANGE_AZ_PASSES, context)
        rearrange_swap_trials = self.parameterAsInt(parameters, self.REARRANGE_SWAP_TRIALS, context)
        rearrange_destroy_size = self.parameterAsInt(parameters, self.REARRANGE_DESTROY_SIZE, context)
        rearrange_destroy_attempts = self.parameterAsInt(parameters, self.REARRANGE_DESTROY_ATTEMPTS, context)
        rearrange_shortlist = self.parameterAsInt(parameters, self.REARRANGE_SHORTLIST, context)
        use_existing_cache = self.parameterAsBool(parameters, self.USE_VIS_CACHE, context)
        visibility_cache_input = self.parameterAsFile(parameters, self.VIS_CACHE_INPUT, context)
        visibility_cache_output = self.parameterAsFileOutput(parameters, self.OUT_VIS_CACHE, context)
        if visibility_cache_output and str(visibility_cache_output).upper() == "TEMPORARY_OUTPUT":
            visibility_cache_output = ""

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

        # Fingerprints protect persistent visibility caches from being reused with
        # a different DEM or a different positive-risk/target mask. FOV, azimuth
        # step and rearrangement settings are intentionally NOT part of the cache.
        dem_hash = self._dem_fingerprint(dem, dem_valid)
        risk_positive_hash = self._risk_positive_fingerprint(risk_flat)

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

        grid_candidate_count_generated = 0
        shoulder_candidate_count_generated = 0
        usable_source_indices = []
        cache = []
        usable_candidates = []
        cache_loaded = False
        cache_saved_path = ""

        if use_existing_cache:
            if candidate_source is not None:
                feedback.pushWarning(
                    "Έχει δοθεί και Candidates layer και visibility cache. Επειδή Use existing cache=ON, "
                    "το Candidates layer και όλα τα candidate-generation settings αγνοούνται."
                )
            (
                candidates, usable_candidates, usable_source_indices, cache,
                candidate_mode_label, cache_meta,
            ) = self._load_visibility_cache(
                visibility_cache_input, rows, cols, gt, projection, observer_height,
                target_height, max_distance, dem_hash, risk_positive_hash, feedback,
            )
            cache_loaded = True
            # Report the actual generation settings saved in the cache, not GUI values
            # that are ignored during a cache-backed run.
            grid_spacing = float(cache_meta.get("grid_spacing_m", grid_spacing))
            terrain_search_step = float(cache_meta.get("terrain_search_step_m", terrain_search_step))
            shoulder_probe_distance = float(cache_meta.get("shoulder_probe_distance_m", shoulder_probe_distance))
            shoulder_max_site_slope = float(cache_meta.get("shoulder_max_site_slope_deg", shoulder_max_site_slope))
            shoulder_min_downhill_slope = float(cache_meta.get("shoulder_min_downhill_slope_deg", shoulder_min_downhill_slope))
            shoulder_max_back_slope = float(cache_meta.get("shoulder_max_back_slope_deg", shoulder_max_back_slope))
            shoulder_min_break = float(cache_meta.get("shoulder_min_terrain_break_deg", shoulder_min_break))
            shoulder_min_spacing = float(cache_meta.get("shoulder_min_spacing_m", shoulder_min_spacing))
            max_shoulder_candidates = int(cache_meta.get("max_shoulder_candidates", max_shoulder_candidates))
            grid_candidate_count_generated = int(cache_meta.get("grid_candidates_generated", 0))
            shoulder_candidate_count_generated = int(cache_meta.get("shoulder_candidates_generated", 0))
            candidate_search_buffer = float(cache_meta.get("target_candidate_buffer_m", candidate_search_buffer))
            candidate_allowed_pixels = int(cache_meta.get("candidate_allowed_buffer_pixels", candidate_allowed_pixels))
        else:
            feedback.pushInfo("Δημιουργία υποψήφιων θέσεων…")
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
                    None, dem_crs, context.transformContext(), gt, rows, cols, dem, dem_valid,
                    landcover, grid_spacing, max_candidates, min_elevation, feedback,
                    candidate_bounds=candidate_bounds, candidate_allowed_mask=candidate_allowed_mask,
                    candidate_type="GRID", apply_max_limit=True,
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
                    None, dem_crs, context.transformContext(), gt, rows, cols, dem, dem_valid,
                    landcover, grid_spacing, max_candidates, min_elevation, feedback,
                    candidate_bounds=candidate_bounds, candidate_allowed_mask=candidate_allowed_mask,
                    candidate_type="GRID", apply_max_limit=False,
                )
                shoulder_candidates = self._build_shoulder_candidates(
                    gt, rows, cols, dem, dem_valid, landcover, min_elevation,
                    terrain_search_step, shoulder_probe_distance,
                    shoulder_max_site_slope, shoulder_min_downhill_slope,
                    shoulder_max_back_slope, shoulder_min_break,
                    shoulder_min_spacing, max_shoulder_candidates, feedback,
                    candidate_bounds=candidate_bounds, candidate_allowed_mask=candidate_allowed_mask,
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
            f"Candidate pool: {len(candidates)} total "
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

        temp_dir = tempfile.mkdtemp(prefix="firecam_viewshed_")

        try:
            if not cache_loaded:
                alg = QgsApplication.processingRegistry().algorithmById("gdal:viewshed")
                if alg is None:
                    raise QgsProcessingException(
                        "Δεν βρέθηκε το gdal:viewshed. Ενεργοποίησε τον GDAL provider στο Processing."
                    )
                parameter_names = {definition.name() for definition in alg.parameterDefinitions()}

                # Use a silent child feedback object for GDAL viewshed calls.
                quiet_viewshed_feedback = QgsProcessingFeedback(False)
                viewshed_log_step = max(25, int(math.ceil(len(candidates) / 20.0)))
                viewshed_started = time.monotonic()
                feedback.pushInfo(
                    f"Viewshed phase: {len(candidates)} candidates. "
                    f"GDAL child logs are suppressed; compact progress will be shown instead."
                )
                for index, candidate in enumerate(candidates):
                    if feedback.isCanceled():
                        break

                    feedback.setProgress(5.0 + 60.0 * index / max(1, len(candidates)))
                    output_path = os.path.join(temp_dir, f"viewshed_{index:05d}.tif")
                    observer_string = f"{candidate['x']:.6f},{candidate['y']:.6f} [{dem_crs.authid()}]"
                    view_params = {
                        "INPUT": dem_layer, "BAND": 1, "OBSERVER": observer_string,
                        "OBSERVER_HEIGHT": observer_height, "TARGET_HEIGHT": target_height,
                        "MAX_DISTANCE": max_distance, "OUTPUT": output_path, "EXTRA": "",
                    }
                    if "CREATION_OPTIONS" in parameter_names:
                        view_params["CREATION_OPTIONS"] = "COMPRESS=LZW|TILED=YES"
                    elif "OPTIONS" in parameter_names:
                        view_params["OPTIONS"] = "COMPRESS=LZW|TILED=YES"

                    try:
                        result = processing.run(
                            "gdal:viewshed", view_params, context=context,
                            feedback=quiet_viewshed_feedback, is_child_algorithm=True,
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
                            (global_rows >= 0) & (global_rows < rows) &
                            (global_cols >= 0) & (global_cols < cols)
                        )
                        global_rows = global_rows[inside]
                        global_cols = global_cols[inside]
                        if global_rows.size == 0:
                            continue

                        flat_indices = (global_rows * cols + global_cols).astype(np.int32)
                        positive_mask = risk_flat[flat_indices] > 0
                        flat_indices = flat_indices[positive_mask]
                        global_rows = global_rows[positive_mask]
                        global_cols = global_cols[positive_mask]
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
                        cache.append({
                            "indices": flat_indices, "bearings": bearings, "quality": quality,
                        })
                        usable_candidates.append(candidate)
                        usable_source_indices.append(index)
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
                        completed_viewsheds == 1 or
                        completed_viewsheds % viewshed_log_step == 0 or
                        completed_viewsheds == len(candidates)
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

                generation_metadata = {
                    "grid_spacing_m": grid_spacing,
                    "grid_candidates_generated": grid_candidate_count_generated,
                    "shoulder_candidates_generated": shoulder_candidate_count_generated,
                    "terrain_search_step_m": terrain_search_step,
                    "shoulder_probe_distance_m": shoulder_probe_distance,
                    "shoulder_max_site_slope_deg": shoulder_max_site_slope,
                    "shoulder_min_downhill_slope_deg": shoulder_min_downhill_slope,
                    "shoulder_max_back_slope_deg": shoulder_max_back_slope,
                    "shoulder_min_terrain_break_deg": shoulder_min_break,
                    "shoulder_min_spacing_m": shoulder_min_spacing,
                    "max_shoulder_candidates": max_shoulder_candidates,
                    "target_candidate_buffer_m": candidate_search_buffer,
                    "candidate_allowed_buffer_pixels": candidate_allowed_pixels,
                }
                if visibility_cache_output:
                    cache_saved_path = self._save_visibility_cache(
                        visibility_cache_output, candidates, usable_source_indices, cache, rows, cols,
                        gt, projection, dem_crs.authid(), observer_height, target_height, max_distance,
                        dem_hash, risk_positive_hash, candidate_mode_label, generation_metadata, feedback,
                    )
            else:
                feedback.setProgress(65.0)
                if visibility_cache_output:
                    in_abs = os.path.abspath(visibility_cache_input)
                    out_abs = os.path.abspath(visibility_cache_output)
                    if not out_abs.lower().endswith(".npz"):
                        out_abs += ".npz"
                    if in_abs != out_abs:
                        os.makedirs(os.path.dirname(out_abs), exist_ok=True)
                        shutil.copy2(in_abs, out_abs)
                        cache_saved_path = out_abs
                        feedback.pushInfo(f"Visibility cache copied to: {out_abs}")
                    else:
                        cache_saved_path = in_abs

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

                    if use_binary_objective:
                        # Fast v8 path: aggregate uncovered risk by integer bearing and
                        # score all azimuths from a 360-bin histogram. This is equivalent
                        # to the old binary mask loop but far cheaper for large runs.
                        orientation = self._orientation_binary_score(
                            item,
                            coverage_count,
                            risk_flat,
                            azimuths,
                            fov,
                            half_fov,
                        )
                        if orientation is not None and orientation["binary_gain"] > 0.0:
                            gain = float(orientation["binary_gain"]) + 1e-6 * float(
                                orientation["quality_tiebreak"]
                            )
                            if gain > best_gain:
                                best_gain = gain
                                best = (candidate_index, orientation)
                    else:
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
                                best = (
                                    candidate_index,
                                    {
                                        "azimuth": float(azimuth),
                                        "indices": selected_indices.copy(),
                                        "quality": selected_quality.copy(),
                                    },
                                )

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

                candidate_index, best_orientation = best
                azimuth = float(best_orientation["azimuth"])
                indices = best_orientation["indices"].copy()
                quality = best_orientation["quality"].copy()
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

            if feedback.isCanceled():
                return {}

            if not selected:
                raise QgsProcessingException("Δεν επιλέχθηκε καμία κάμερα.")

            # Scalar snapshot of the ORIGINAL greedy solution. This remains untouched
            # even if the v8 rearrangement later swaps/removes cameras.
            initial_selected = []
            for camera in selected:
                camera["greedy_rank"] = int(camera["rank"])
                camera["rearranged"] = 0
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

            greedy_selected_camera_count = len(selected)
            greedy_system_quality, greedy_system_count = self._compose_coverage(
                selected, rows * cols
            )
            initial_system_score = float(
                np.sum(risk_flat * greedy_system_quality, dtype=np.float64)
            )
            initial_binary_after_greedy = 100.0 * float(
                np.sum(risk_flat[greedy_system_count > 0], dtype=np.float64) / total_risk
            )

            rearrangement_stats = {
                "azimuth_changes": 0,
                "one_swaps": 0,
                "destroy_repairs": 0,
                "cameras_removed": 0,
                "passes_completed": 0,
                "final_camera_count": len(selected),
                "final_binary_pct": initial_binary_after_greedy,
            }

            if rearrange_enable:
                if use_binary_objective:
                    feedback.setProgress(90.0)
                    selected, rearrangement_stats = self._local_rearrange_binary(
                        selected,
                        cache,
                        usable_candidates,
                        dem,
                        risk_flat,
                        total_risk,
                        target_coverage,
                        azimuths,
                        fov,
                        half_fov,
                        rearrange_max_passes,
                        rearrange_az_passes,
                        rearrange_swap_trials,
                        rearrange_destroy_size,
                        rearrange_destroy_attempts,
                        rearrange_shortlist,
                        feedback,
                    )
                    if feedback.isCanceled():
                        return {}
                else:
                    feedback.pushWarning(
                        "Η v8 rearrangement φάση είναι binary-objective local search. "
                        "Παραλείπεται επειδή έχει επιλεγεί Quality coverage objective."
                    )

            post_rearrangement_quality, post_rearrangement_count = self._compose_coverage(
                selected, rows * cols
            )
            post_rearrangement_system_score = float(
                np.sum(risk_flat * post_rearrangement_quality, dtype=np.float64)
            )
            post_rearrangement_binary = 100.0 * float(
                np.sum(risk_flat[post_rearrangement_count > 0], dtype=np.float64) / total_risk
            )
            target_coverage_reached_after_rearrange = int(
                use_target_coverage
                and post_rearrangement_binary + 1e-9 >= target_coverage
            )

            # No separate uphill refinement in v8.1. The rearranged solution is final.
            final_selected = []
            for rank, camera in enumerate(selected, start=1):
                copied = dict(camera)
                copied["rank"] = rank
                copied["indices"] = camera["indices"].copy()
                copied["quality"] = camera["quality"].copy()
                copied.setdefault("greedy_rank", 0)
                copied.setdefault("rearranged", 0)
                final_selected.append(copied)

            # Recompute final system coverage from the final rearranged camera set.
            current_quality, coverage_count = self._compose_coverage(
                final_selected, rows * cols
            )

            # Recompute marginal metrics in rank order for the final positions.
            running_quality = np.zeros(rows * cols, dtype=np.float32)
            running_count = np.zeros(rows * cols, dtype=np.uint16)
            previous_binary_pct = 0.0
            for camera in final_selected:
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
                camera["delta_binary_pct"] = camera["binary_pct"] - previous_binary_pct
                previous_binary_pct = camera["binary_pct"]

            # ---------- Vector outputs: original greedy and final rearranged solution ----------
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
            final_fields.append(QgsField("delta_bin", QVariant.Double))
            final_fields.append(QgsField("greedy_rk", QVariant.Int))
            final_fields.append(QgsField("rearrang", QVariant.Int))
            final_fields.append(QgsField("obs_h_m", QVariant.Double))
            final_fields.append(QgsField("target_h_m", QVariant.Double))
            final_fields.append(QgsField("range_m", QVariant.Double))
            final_fields.append(QgsField("elev_m", QVariant.Double))
            final_fields.append(QgsField("cand_type", QVariant.String))
            final_fields.append(QgsField("sh_score", QVariant.Double))
            final_fields.append(QgsField("sh_az", QVariant.Double))
            final_fields.append(QgsField("down_slp", QVariant.Double))
            final_fields.append(QgsField("back_slp", QVariant.Double))
            final_fields.append(QgsField("break_deg", QVariant.Double))
            final_fields.append(QgsField("site_slp", QVariant.Double))

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
            fov_sink, fov_dest = self.parameterAsSink(
                parameters,
                self.OUT_FOV,
                context,
                final_fields,
                QgsWkbTypes.Polygon,
                dem_crs,
            )
            if initial_sink is None or camera_sink is None or fov_sink is None:
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

            for camera in final_selected:
                attributes = [
                    camera["rank"],
                    camera["azimuth"],
                    fov,
                    camera["gain"],
                    camera["effective_pct"],
                    camera["binary_pct"],
                    camera.get("delta_binary_pct", 0.0),
                    int(camera.get("greedy_rank", 0)),
                    int(camera.get("rearranged", 0)),
                    observer_height,
                    target_height,
                    max_distance,
                    camera["elev_m"],
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

            # ---------- Raster outputs based on FINAL rearranged positions ----------
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
            selected_grid_count = sum(1 for c in final_selected if c.get("candidate_type") == "GRID")
            selected_shoulder_count = sum(1 for c in final_selected if c.get("candidate_type") == "SHOULDER")
            selected_supplied_count = sum(1 for c in final_selected if c.get("candidate_type") == "SUPPLIED")
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
                writer.writerow(["visibility_cache_loaded", int(cache_loaded)])
                writer.writerow(["visibility_cache_input", visibility_cache_input if cache_loaded else ""])
                writer.writerow(["visibility_cache_saved", int(bool(cache_saved_path))])
                writer.writerow(["visibility_cache_output", cache_saved_path])
                writer.writerow(["visibility_cache_format_version", 2])
                writer.writerow(["selection_mode", "target binary coverage" if use_target_coverage else "fixed camera count"])
                writer.writerow(["selection_objective", selection_objective_label])
                writer.writerow(["marginal_gain_meaning", "weighted newly covered risk plus tiny quality tie-break" if use_binary_objective else "weighted quality improvement"])
                writer.writerow(["max_camera_count", num_cameras])
                writer.writerow(["rearrangement_enabled", int(bool(rearrange_enable and use_binary_objective))])
                writer.writerow(["rearrangement_max_passes", rearrange_max_passes])
                writer.writerow(["rearrangement_azimuth_passes", rearrange_az_passes])
                writer.writerow(["rearrangement_swap_trials", rearrange_swap_trials])
                writer.writerow(["rearrangement_destroy_size", rearrange_destroy_size])
                writer.writerow(["rearrangement_destroy_attempts", rearrange_destroy_attempts])
                writer.writerow(["rearrangement_destroy_group_strategy", "mixed_overlap_spatial_weak_v1"])
                writer.writerow(["rearrangement_candidate_shortlist", rearrange_shortlist])
                writer.writerow(["greedy_selected_camera_count", greedy_selected_camera_count])
                writer.writerow(["post_rearrangement_camera_count", len(selected)])
                writer.writerow(["rearrangement_azimuth_changes", rearrangement_stats.get("azimuth_changes", 0)])
                writer.writerow(["rearrangement_one_swaps", rearrangement_stats.get("one_swaps", 0)])
                writer.writerow(["rearrangement_destroy_repairs", rearrangement_stats.get("destroy_repairs", 0)])
                writer.writerow(["rearrangement_cameras_removed", rearrangement_stats.get("cameras_removed", 0)])
                writer.writerow(["rearrangement_passes_completed", rearrangement_stats.get("passes_completed", 0)])
                writer.writerow(["selected_camera_count", len(final_selected)])
                writer.writerow(["selected_grid_camera_count", selected_grid_count])
                writer.writerow(["selected_shoulder_camera_count", selected_shoulder_count])
                writer.writerow(["selected_supplied_camera_count", selected_supplied_count])
                writer.writerow(["requested_binary_coverage_pct", target_coverage])
                writer.writerow(["target_coverage_reached_after_greedy", int(target_coverage_reached_initial)])
                writer.writerow(["target_coverage_reached_at_rank", target_coverage_reached_rank])
                writer.writerow(["target_coverage_reached_after_rearrangement", target_coverage_reached_after_rearrange])
                writer.writerow(["target_coverage_reached_final", final_target_coverage_reached])
                writer.writerow(["initial_binary_after_greedy_pct", initial_binary_after_greedy])
                writer.writerow(["post_rearrangement_binary_pct", post_rearrangement_binary])
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
                writer.writerow(["initial_system_score", initial_system_score])
                writer.writerow(["post_rearrangement_system_score", post_rearrangement_system_score])
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
                        "x",
                        "y",
                        "elev_m",
                        "azimuth_deg",
                        "greedy_rank",
                        "rearranged",
                        "candidate_type",
                        "shoulder_score",
                        "shoulder_azimuth_deg",
                        "downhill_slope_deg",
                        "back_slope_deg",
                        "terrain_break_deg",
                        "site_slope_deg",
                        "marginal_gain",
                        "effective_pct",
                        "binary_pct",
                        "delta_binary_pct",
                    ]
                )
                for camera in final_selected:
                    writer.writerow(
                        [
                            camera["rank"],
                            camera["x"],
                            camera["y"],
                            camera["elev_m"],
                            camera["azimuth"],
                            int(camera.get("greedy_rank", 0)),
                            int(camera.get("rearranged", 0)),
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
                            camera.get("delta_binary_pct", 0.0),
                        ]
                    )

            feedback.pushInfo(f"CSV summary saved at: {summary_path}")
            feedback.setProgress(100.0)
            feedback.pushInfo(
                f"Ολοκληρώθηκε: greedy {greedy_selected_camera_count} → final {len(final_selected)} κάμερες "
                f"(GRID={selected_grid_count}, SHOULDER={selected_shoulder_count}, SUPPLIED={selected_supplied_count}), "
                f"τελική eff_pct {final_effective:.2f}%, τελική bin_pct {final_binary:.2f}%, "
                f"forest class 10 bin_pct {forest_bin_pct:.2f}%, "
                f"τελικό score {100.0 * final_system_score / max(initial_system_score, 1e-12):.2f}% "
                f"του αρχικού greedy score."
            )
            if use_target_coverage:
                if final_binary + 1e-9 >= target_coverage:
                    feedback.pushInfo(
                        f"Ο στόχος bin_pct {target_coverage:.2f}% επιτεύχθηκε με {len(final_selected)} κάμερες."
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
                self.OUT_FOV: fov_dest,
                self.OUT_QUALITY: output_quality,
                self.OUT_COUNT: output_count,
                self.OUT_REMAINING_RISK: output_remaining,
                self.OUT_UNCOVERED_TARGET: output_uncovered_target,
                self.OUT_UNCOVERED_FOREST: output_uncovered_forest,
                self.OUT_SUMMARY: summary_path,
                self.OUT_VIS_CACHE: cache_saved_path if cache_saved_path else (visibility_cache_input if cache_loaded else ""),
            }

        finally:
            dem_ds = None
            lc_ds = None
            shutil.rmtree(temp_dir, ignore_errors=True)
