#!/usr/bin/env python3

import os
import argparse
import pathlib

import acts
import acts.examples
from acts.examples.simulation import (
    addParticleGun,
    MomentumConfig,
    EtaConfig,
    PhiConfig,
    ParticleConfig,
    addPythia8,
    addFatras,
    addGeant4,
    ParticleSelectorConfig,
    addDigitization,
    addParticleSelection,
)

from acts.examples.reconstruction import (
    addSeeding,
    TruthSeedRanges,
    addCKFTracks,
    TrackSelectorConfig,
    addAmbiguityResolution,
    AmbiguityResolutionConfig,
    addAmbiguityResolutionML,
    AmbiguityResolutionMLConfig,
    addVertexFitting,
    VertexFinder,
    addSeedPerformanceWriters,
    addSeedingTruthSelection,
    addSpacePointsMaking,
)

import acts.examples.reconstruction as reconstruction

from collections import namedtuple
from pathlib import Path

from typing import Optional, Union
from enum import Enum

eta = 4
# eta = 2.5

SeedingAlgorithm = Enum(
    "SeedingAlgorithm", "Default TruthSmeared TruthEstimated Orthogonal HashingSeeding"
)

DetectorName = Enum("DetectorName", "ODD generic ITk")

SeedFinderConfigName = Enum("SeedFinderConfigName", "TrackML cpp")

HashingMetric = Enum("HashingMetric", "dphi dR")

parser = argparse.ArgumentParser()
parser.add_argument("--mu", type=int, default=0)
parser.add_argument("--bucketSize", type=int, default=0)
parser.add_argument(
    "--nevents",
    type=int,
    # default=1000)
    default=100,
)
parser.add_argument("--maxSeedsPerSpM", type=int, default=1)
parser.add_argument("--seedingAlgorithm", type=str)
parser.add_argument("--saveFilesSmall", type=bool, default=False)
parser.add_argument("--saveFiles", type=bool, default=False)
parser.add_argument("--AnnoySeed", type=int, default=123456789)
parser.add_argument("--zBins", type=int, default=0)
parser.add_argument("--phiBins", type=int, default=0)
parser.add_argument("--metric", type=str)
args = parser.parse_args()

print(args)

mu = args.mu
bucketSize = args.bucketSize
nevents = args.nevents
saveFiles = args.saveFiles
AnnoySeed = args.AnnoySeed
zBins = args.zBins
phiBins = args.phiBins
maxSeedsPerSpM = args.maxSeedsPerSpM

seedingAlgorithm = SeedingAlgorithm.HashingSeeding
if args.seedingAlgorithm:
    if args.seedingAlgorithm == "Default":
        seedingAlgorithm = SeedingAlgorithm.Default
    elif args.seedingAlgorithm == "Orthogonal":
        seedingAlgorithm = SeedingAlgorithm.Orthogonal
    elif args.seedingAlgorithm == "HashingSeeding":
        seedingAlgorithm = SeedingAlgorithm.HashingSeeding

metric = HashingMetric.dphi
if args.metric:
    if args.metric == "dphi":
        metric = HashingMetric.dphi
    elif args.metric == "dR":
        metric = HashingMetric.dR

if seedingAlgorithm == SeedingAlgorithm.Default:
    bucketSize = 0
    metric = HashingMetric.dphi
    AnnoySeed = 123456789
    zBins = 0
    phiBins = 0

print(mu, bucketSize, seedingAlgorithm)


def extractEnumName(enumvar):
    return str(enumvar).split(".")[-1]


u = acts.UnitConstants


def getActsExamplesDirectory():
    return Path(__file__).parent.parent.parent


Config = namedtuple(
    "Config",
    [
        "mu",
        "bucketSize",
        "maxSeedsPerSpM",
        "seedFinderConfig",
        "detector",
        "seedingAlgorithm",
        "metric",
        "AnnoySeed",
        "zBins",
        "phiBins",
    ],
    defaults=[
        None,
        0,
        1,
        SeedFinderConfigName.TrackML,
        DetectorName.generic,
        SeedingAlgorithm.HashingSeeding,
        "angular",
        123456789,
        0,
        0,
    ],
)
# https://stackoverflow.com/questions/34269772/type-hints-in-namedtuple
Config.__annotations__ = {
    "mu": int,
    "bucketSize": int,
    "maxSeedsPerSpM": int,
    "seedFinderConfig": SeedFinderConfigName,
    "detector": DetectorName,
    "seedingAlgorithm": SeedingAlgorithm,
    "metric": str,
    "AnnoySeed": int,
    "zBins": int,
    "phiBins": int,
}

# config = Config(mu=50, bucketSize=0, maxSeedsPerSpM=5, seedFinderConfig="cpp", detector=DetectorName.ODD)
# config = Config(mu=mu, bucketSize=bucketSize, maxSeedsPerSpM=1, seedFinderConfig="TrackML", detector=DetectorName.generic, seedingAlgorithm=SeedingAlgorithm.HashingSeeding)
config = Config(
    mu=mu,
    bucketSize=bucketSize,
    maxSeedsPerSpM=maxSeedsPerSpM,
    seedFinderConfig="TrackML",
    detector=DetectorName.generic,
    # detector=DetectorName.ODD,
    # config = Config(mu=mu, bucketSize=bucketSize, maxSeedsPerSpM=maxSeedsPerSpM, seedFinderConfig="TrackML", detector=DetectorName.ODD,
    seedingAlgorithm=seedingAlgorithm,
    metric=metric,
    AnnoySeed=AnnoySeed,
    zBins=zBins,
    phiBins=phiBins,
)
# config = Config(mu=mu, bucketSize=bucketSize, maxSeedsPerSpM=1, seedFinderConfig="TrackML", detector=DetectorName.ODD,
#                 seedingAlgorithm=seedingAlgorithm, metric=metric, AnnoySeed=AnnoySeed, zBins=zBins)
# config = Config(mu=mu, bucketSize=bucketSize, maxSeedsPerSpM=1, seedFinderConfig="TrackML", detector=DetectorName.generic, seedingAlgorithm=seedingAlgorithm)

# config = Config(mu=mu, bucketSize=bucketSize, maxSeedsPerSpM=1000, seedFinderConfig="TrackML", detector=DetectorName.generic,
#                 seedingAlgorithm=seedingAlgorithm, metric=metric, AnnoySeed=AnnoySeed, zBins=zBins, phiBins=phiBins)

actsExamplesDir = getActsExamplesDirectory()

if config.detector == DetectorName.ODD:
    from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory

    geoDir = getOpenDataDetectorDirectory()

    oddMaterialMap = geoDir / "data/odd-material-maps.root"
    oddDigiConfig = geoDir / "config/odd-digi-smearing-config.json"
    oddSeedingSel = geoDir / "config/odd-seeding-config.json"
    oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)

    detector, trackingGeometry, decorators = getOpenDataDetector(
        odd_dir=geoDir, mdecorator=oddMaterialDeco
    )

    digiConfig = oddDigiConfig

    # geoSelectionConfigFile = actsExamplesDir / "Algorithms/TrackFinding/share/geoSelection-openDataDetector.json"
    geoSelectionConfigFile = oddSeedingSel

elif config.detector == DetectorName.generic:
    print("Create detector and tracking geometry")

    detector, trackingGeometry, _ = acts.examples.GenericDetector.create()
    digiConfig = (
        actsExamplesDir
        / "Algorithms/Digitization/share/default-smearing-config-generic.json"
    )
    geoSelectionConfigFile = (
        actsExamplesDir
        / "Algorithms/TrackFinding/share/geoSelection-genericDetector.json"
    )
elif config.detector == DetectorName.ITk:
    geo_dir = actsExamplesDir.parent.parent / "acts-itk"

    detector, trackingGeometry, decorators = acts.examples.itk.buildITkGeometry(geo_dir)

    digiConfig = geo_dir / "itk-hgtd/itk-smearing-config.json"

    geoSelectionConfigFile = geo_dir / "itk-hgtd/geoSelection-ITk.json"

    field = acts.examples.MagneticFieldMapXyz(
        str(geo_dir / "bfield/ATLAS-BField-xyz.root")
    )
else:
    exit("Detector not supported")

truthSeedRanges = TruthSeedRanges(
    pt=(1.0 * u.GeV, None), eta=(-eta, eta), nHits=(9, None)
)

CKFptMin = 1.0 * u.GeV

doHashing = config.bucketSize > 0
bucketSize = config.bucketSize
npileup = config.mu
maxSeedsPerSpM = config.maxSeedsPerSpM


def get_dir_config(config: Config):
    global main_dir
    outDir = f"detector_{extractEnumName(config.detector)}"
    outDir += "_output"
    doHashing = config.bucketSize > 0
    if doHashing:
        outDir += "_hashing"

    outDir += f"_mu_{config.mu}"
    if doHashing:
        outDir += f"_bucket_{config.bucketSize}"

    outDir += f"_maxSeedsPerSpM_{config.maxSeedsPerSpM}"

    outDir += f"_seedFinderConfig_{extractEnumName(config.seedFinderConfig)}"

    outDir += f"_seedingAlgorithm_{extractEnumName(config.seedingAlgorithm)}"
    if doHashing:
        if config.metric != "angular":
            outDir += f"_metric_{extractEnumName(config.metric)}"
        outDir += f"_AnnoySeed_{config.AnnoySeed}"
        if config.zBins != 0:
            outDir += f"_zBins_{config.zBins}"
        if config.phiBins != 0:
            outDir += f"_phiBins_{config.phiBins}"
    return outDir


outDir = get_dir_config(config)
# outDir += "_binned merge_1"
# outDir += f"_eta_{eta}"
print(outDir)

outputDir = pathlib.Path.cwd() / outDir

if not outputDir.exists():
    outputDir.mkdir(parents=True)

config_file = open(outputDir / "config_file.txt", "w")
config_file.write(str(config))
config_file.close()

# acts.examples.dump_args_calls(locals())  # show python binding calls

if config.detector != DetectorName.ITk:
    field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
else:
    exit(f"Detector not supported {config.detector}")
rnd = acts.examples.RandomNumbers(seed=42)

s = acts.examples.Sequencer(
    events=nevents,
    numThreads=1,
    outputDir=str(outputDir),
    trackFpes=False,
    enableEventTiming=True,
)

addPythia8(
    s,
    hardProcess=["Top:qqbar2ttbar=on"],
    npileup=npileup,
    vtxGen=acts.examples.GaussianVertexGenerator(
        # stddev=acts.Vector4(0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns),
        stddev=acts.Vector4(0, 0, 50 * u.mm, 0),
        mean=acts.Vector4(0, 0, 0, 0),
    ),
    rnd=rnd,
    # outputDirRoot=outputDir,
    # outputDirCsv=outputDir if saveFiles else None,
)

g4_simulation = False

if g4_simulation:
    if s.config.numThreads != 1:
        raise ValueError("Geant 4 simulation does not support multi-threading")

    # Pythia can sometime simulate particles outside the world volume, a cut on the Z of the track help mitigate this effect
    # Older version of G4 might not work, this as has been tested on version `geant4-11-00-patch-03`
    # For more detail see issue #1578
    addGeant4(
        s,
        detector,
        trackingGeometry,
        field,
        # g4DetectorConstructionFactory=DDG4DetectorConstructionFactory(detector),
        # g4DetectorConstructionFactory=TelescopeG4DetectorConstructionFactory(detector),
        # g4DetectorConstructionFactory=detector,
        preSelectParticles=ParticleSelectorConfig(
            rho=(0.0, 28 * u.mm),
            absZ=(0.0, 1.0 * u.m),
            eta=(-4.0, 4.0),
            pt=(150 * u.MeV, None),
            removeNeutral=True,
        ),
        outputDirRoot=outputDir,
        # outputDirCsv=outputDir,
        rnd=rnd,
        killVolume=trackingGeometry.worldVolume,
        killAfterTime=25 * u.ns,
    )
else:
    addFatras(
        s,
        trackingGeometry,
        field,
        preSelectParticles=ParticleSelectorConfig(
            eta=(-eta, eta), pt=(150 * u.MeV, None), removeNeutral=True
        ),
        enableInteractions=True,
        outputDirRoot=outputDir,
        outputDirCsv=outputDir if saveFiles else None,
        rnd=rnd,
    )

addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=digiConfig,
    outputDirRoot=outputDir,
    outputDirCsv=outputDir if saveFiles else None,
    rnd=rnd,
)


# logLevel = acts.logging.DEBUG
# s.config.logLevel = acts.logging.DEBUG

# ParticleSmearingSigmas = namedtuple(
#     "ParticleSmearingSigmas",
#     ["d0", "d0PtA", "d0PtB", "z0", "z0PtA", "z0PtB", "t0", "phi", "theta", "pRel"],
#     defaults=[None] * 10,
# )


# SeedFinderConfigArg = namedtuple(
#     "SeedFinderConfig",
#     [
#         "maxSeedsPerSpM",
#         "cotThetaMax",
#         "sigmaScattering",
#         "radLengthPerSeed",
#         "minPt",
#         "impactMax",
#         "interactionPointCut",
#         "arithmeticAverageCotTheta",
#         "deltaZMax",
#         "maxPtScattering",
#         "zBinEdges",
#         "skipPreviousTopSP",
#         "zBinsCustomLooping",
#         "rRangeMiddleSP",
#         "useVariableMiddleSPRange",
#         "binSizeR",
#         "forceRadialSorting",
#         "seedConfirmation",
#         "centralSeedConfirmationRange",
#         "forwardSeedConfirmationRange",
#         "deltaR",  # (min,max)
#         "deltaRBottomSP",  # (min,max)
#         "deltaRTopSP",  # (min,max)
#         "deltaRMiddleSPRange",  # (min,max)
#         "collisionRegion",  # (min,max)
#         "r",  # (min,max)
#         "z",  # (min,max)
#     ],
#     defaults=[None] * 20 + [(None, None)] * 7,
# )

SeedFinderConfigArg = reconstruction.SeedFinderConfigArg
SeedFinderOptionsArg = reconstruction.SeedFinderOptionsArg
SeedFilterConfigArg = reconstruction.SeedFilterConfigArg
SpacePointGridConfigArg = reconstruction.SpacePointGridConfigArg
SeedingAlgorithmConfigArg = reconstruction.SeedingAlgorithmConfigArg


import numpy as np

cotThetaMax = 1 / (np.tan(2 * np.arctan(np.exp(-eta))))  # =1/tan(2×atan(e^(-eta)))
# Issue for TrackML and CKF for cotThetaMax with eta > 3.25 with commit d9f775f4155f1a13e316aa142e939ef7178ff665
if config.seedFinderConfig == "TrackML":
    seedFinderConfigArg = SeedFinderConfigArg(
        r=(None, 200 * u.mm),  # rMin=default, 33mm
        deltaR=(1 * u.mm, 60 * u.mm),
        collisionRegion=(-250 * u.mm, 250 * u.mm),
        z=(-2000 * u.mm, 2000 * u.mm),
        maxSeedsPerSpM=maxSeedsPerSpM,
        sigmaScattering=5,
        radLengthPerSeed=0.1,
        minPt=500 * u.MeV,
        impactMax=3 * u.mm,
        cotThetaMax=cotThetaMax,  # =1/tan(2×atan(e^(-eta)))
        # cotThetaMax = 1000, # Hashing better perfs with that; in SPGrid: float zBinSize = config.cotThetaMax * config.deltaRMax;
        # int zBins = max(1, (int)std::floor((config.zMax - config.zMin) / zBinSize))
    )
elif config.seedFinderConfig == "cpp":
    seedFinderConfigArg = SeedFinderConfigArg(
        maxSeedsPerSpM=maxSeedsPerSpM, cotThetaMax=cotThetaMax
    )
else:
    exit("seedFinderConfig not supported")

seedFinderOptionsArg: SeedFinderOptionsArg = SeedFinderOptionsArg(
    bFieldInZ=1.99724 * u.T
)
seedFilterConfigArg: SeedFilterConfigArg = SeedFilterConfigArg()
spacePointGridConfigArg: SpacePointGridConfigArg = SpacePointGridConfigArg()
seedingAlgorithmConfigArg: SeedingAlgorithmConfigArg = SeedingAlgorithmConfigArg()
inputParticles: str = "particles"
outputDirRoot: Optional[Union[Path, str]] = outputDir
logLevel: Optional[acts.logging.Level] = None

logLevel = acts.examples.defaultLogging(s, logLevel)()
customLogLevel = acts.examples.defaultLogging(s, logLevel)
logger = acts.logging.getLogger("addSeeding")

if truthSeedRanges is not None:
    selectedParticles = "truth_seeds_selected"
    addSeedingTruthSelection(
        s,
        inputParticles,
        selectedParticles,
        truthSeedRanges,
        logLevel,
        # level=acts.logging.INFO,
    )
else:
    selectedParticles = inputParticles

print(selectedParticles)

# Create starting parameters from either particle smearing or combined seed
# finding and track parameters estimation
spacePoints = addSpacePointsMaking(
    s, trackingGeometry, geoSelectionConfigFile, logLevel
)

outputDirRoot = Path(outputDirRoot)
if not outputDirRoot.exists():
    outputDirRoot.mkdir()
s.addWriter(
    acts.examples.RootSpacepointWriter(
        level=customLogLevel(),
        inputSpacepoints=spacePoints,
        filePath=str(outputDirRoot / "spacepoints.root"),
    )
)

# s.config.logLevel = acts.logging.VERBOSE

from typing import Optional, Union, List


def addHashing(
    bucketSize: Optional[int] = 10,
    AnnoySeed: Optional[int] = 123456789,
    zBins: Optional[int] = 0,
    phiBins: Optional[int] = 100,
    metric=HashingMetric.dphi,
) -> acts.examples.Sequencer:
    """This function steers the digitization step

    Parameters
    ----------
    s: Sequencer
        the sequencer module to which we add the Digitization steps (returned from addDigitization)
    outputDirCsv : Path|str, path, None
        the output folder for the Csv output, None triggers no output
    outputDirRoot : Path|str, path, None
        the output folder for the Root output, None triggers no output

    nBucketsLimit : Optional[int]
        superior limit on the total number of buckets
    """

    if int(s.config.logLevel) <= int(acts.logging.DEBUG):
        acts.examples.dump_args_calls(locals())

    f = 1
    if metric == HashingMetric.dphi:
        f = 1
    elif metric == HashingMetric.dR:
        f = 2

    # Hashing
    hashingTrainingCfg = acts.HashingTrainingConfig(
        AnnoySeed=AnnoySeed,
        f=f,
    )

    hashingCfg = acts.HashingAlgorithmConfig(
        bucketSize=bucketSize,
        zBins=zBins,
        phiBins=phiBins,
    )

    return hashingTrainingCfg, hashingCfg


if doHashing:
    # for now hashing only use space points and not clusters
    print("***> addHashing")
    hashingTrainingCfg, hashingCfg = addHashing(
        bucketSize=bucketSize,
        AnnoySeed=AnnoySeed,
        zBins=zBins,
        phiBins=phiBins,
        metric=config.metric,
    )

# From addStandardSeeding
seedFinderConfig = acts.SeedFinderConfig(
    **acts.examples.defaultKWArgs(
        rMin=seedFinderConfigArg.r[0],
        rMax=seedFinderConfigArg.r[1],
        deltaRMin=seedFinderConfigArg.deltaR[0],
        deltaRMax=seedFinderConfigArg.deltaR[1],
        deltaRMinTopSP=(
            seedFinderConfigArg.deltaR[0]
            if seedFinderConfigArg.deltaRTopSP[0] is None
            else seedFinderConfigArg.deltaRTopSP[0]
        ),
        deltaRMaxTopSP=(
            seedFinderConfigArg.deltaR[1]
            if seedFinderConfigArg.deltaRTopSP[1] is None
            else seedFinderConfigArg.deltaRTopSP[1]
        ),
        deltaRMinBottomSP=(
            seedFinderConfigArg.deltaR[0]
            if seedFinderConfigArg.deltaRBottomSP[0] is None
            else seedFinderConfigArg.deltaRBottomSP[0]
        ),
        deltaRMaxBottomSP=(
            seedFinderConfigArg.deltaR[1]
            if seedFinderConfigArg.deltaRBottomSP[1] is None
            else seedFinderConfigArg.deltaRBottomSP[1]
        ),
        deltaRMiddleMinSPRange=seedFinderConfigArg.deltaRMiddleSPRange[0],
        deltaRMiddleMaxSPRange=seedFinderConfigArg.deltaRMiddleSPRange[1],
        collisionRegionMin=seedFinderConfigArg.collisionRegion[0],
        collisionRegionMax=seedFinderConfigArg.collisionRegion[1],
        zMin=seedFinderConfigArg.z[0],
        zMax=seedFinderConfigArg.z[1],
        zOutermostLayers=(
            (
                seedFinderConfigArg.zOutermostLayers[0]
                if seedFinderConfigArg.zOutermostLayers[0] is not None
                else seedFinderConfigArg.z[0]
            ),
            (
                seedFinderConfigArg.zOutermostLayers[1]
                if seedFinderConfigArg.zOutermostLayers[1] is not None
                else seedFinderConfigArg.z[1]
            ),
        ),
        maxSeedsPerSpM=seedFinderConfigArg.maxSeedsPerSpM,
        cotThetaMax=seedFinderConfigArg.cotThetaMax,
        sigmaScattering=seedFinderConfigArg.sigmaScattering,
        radLengthPerSeed=seedFinderConfigArg.radLengthPerSeed,
        minPt=seedFinderConfigArg.minPt,
        impactMax=seedFinderConfigArg.impactMax,
        interactionPointCut=seedFinderConfigArg.interactionPointCut,
        deltaZMax=seedFinderConfigArg.deltaZMax,
        maxPtScattering=seedFinderConfigArg.maxPtScattering,
        zBinEdges=seedFinderConfigArg.zBinEdges,
        zBinsCustomLooping=seedFinderConfigArg.zBinsCustomLooping,
        rRangeMiddleSP=seedFinderConfigArg.rRangeMiddleSP,
        useVariableMiddleSPRange=seedFinderConfigArg.useVariableMiddleSPRange,
        binSizeR=seedFinderConfigArg.binSizeR,
        seedConfirmation=seedFinderConfigArg.seedConfirmation,
        centralSeedConfirmationRange=seedFinderConfigArg.centralSeedConfirmationRange,
        forwardSeedConfirmationRange=seedFinderConfigArg.forwardSeedConfirmationRange,
    ),
)
seedFinderOptions = acts.SeedFinderOptions(
    **acts.examples.defaultKWArgs(
        beamPos=(
            acts.Vector2(0.0, 0.0)
            if seedFinderOptionsArg.beamPos == (None, None)
            else acts.Vector2(
                seedFinderOptionsArg.beamPos[0], seedFinderOptionsArg.beamPos[1]
            )
        ),
        bFieldInZ=seedFinderOptionsArg.bFieldInZ,
    )
)
seedFilterConfig = acts.SeedFilterConfig(
    **acts.examples.defaultKWArgs(
        maxSeedsPerSpM=seedFinderConfig.maxSeedsPerSpM,
        deltaRMin=(
            seedFinderConfig.deltaRMin
            if seedFilterConfigArg.deltaRMin is None
            else seedFilterConfigArg.deltaRMin
        ),
        impactWeightFactor=seedFilterConfigArg.impactWeightFactor,
        zOriginWeightFactor=seedFilterConfigArg.zOriginWeightFactor,
        compatSeedWeight=seedFilterConfigArg.compatSeedWeight,
        compatSeedLimit=seedFilterConfigArg.compatSeedLimit,
        numSeedIncrement=seedFilterConfigArg.numSeedIncrement,
        seedWeightIncrement=seedFilterConfigArg.seedWeightIncrement,
        seedConfirmation=seedFilterConfigArg.seedConfirmation,
        centralSeedConfirmationRange=seedFinderConfig.centralSeedConfirmationRange,
        forwardSeedConfirmationRange=seedFinderConfig.forwardSeedConfirmationRange,
        maxSeedsPerSpMConf=seedFilterConfigArg.maxSeedsPerSpMConf,
        maxQualitySeedsPerSpMConf=seedFilterConfigArg.maxQualitySeedsPerSpMConf,
        useDeltaRorTopRadius=seedFilterConfigArg.useDeltaRorTopRadius,
    )
)

gridConfig = acts.SpacePointGridConfig(
    **acts.examples.defaultKWArgs(
        minPt=seedFinderConfig.minPt,
        rMax=(
            seedFinderConfig.rMax
            if spacePointGridConfigArg.rMax is None
            else spacePointGridConfigArg.rMax
        ),
        zMax=seedFinderConfig.zMax,
        zMin=seedFinderConfig.zMin,
        deltaRMax=(
            seedFinderConfig.deltaRMax
            if spacePointGridConfigArg.deltaRMax is None
            else spacePointGridConfigArg.deltaRMax
        ),
        cotThetaMax=seedFinderConfig.cotThetaMax,
        phiMin=spacePointGridConfigArg.phi[0],
        phiMax=spacePointGridConfigArg.phi[1],
        maxPhiBins=spacePointGridConfigArg.maxPhiBins,
        impactMax=spacePointGridConfigArg.impactMax,
        zBinEdges=spacePointGridConfigArg.zBinEdges,
        phiBinDeflectionCoverage=spacePointGridConfigArg.phiBinDeflectionCoverage,
    )
)

gridOptions = acts.SpacePointGridOptions(
    **acts.examples.defaultKWArgs(
        bFieldInZ=seedFinderOptions.bFieldInZ,
    )
)

if config.seedingAlgorithm == SeedingAlgorithm.Default:
    logger.info("Using default seeding")
    # logLevel = acts.examples.defaultLogging(s, logLevel)()

    seedingAlg = acts.examples.SeedingAlgorithm(
        level=logLevel,
        inputSpacePoints=[spacePoints],
        outputSeeds="seeds",
        **acts.examples.defaultKWArgs(
            allowSeparateRMax=seedingAlgorithmConfigArg.allowSeparateRMax,
            zBinNeighborsTop=seedingAlgorithmConfigArg.zBinNeighborsTop,
            zBinNeighborsBottom=seedingAlgorithmConfigArg.zBinNeighborsBottom,
            numPhiNeighbors=seedingAlgorithmConfigArg.numPhiNeighbors,
        ),
        gridConfig=gridConfig,
        gridOptions=gridOptions,
        seedFilterConfig=seedFilterConfig,
        seedFinderConfig=seedFinderConfig,
        seedFinderOptions=seedFinderOptions,
    )
    s.addAlgorithm(seedingAlg)

elif config.seedingAlgorithm == SeedingAlgorithm.HashingSeeding:
    # assert(doHashing)
    # if doHashing:
    #     bucketSP = "hashingSPBucket_{}".format(0)
    # else:
    #     bucketSP = "spacepoints"
    logger.info("Using Hashing seeding")

    seedingAlg = acts.examples.SeedingAlgorithmHashing(
        level=logLevel,
        inputSpacePoints=["spacepoints"],
        outputSeeds="seeds",
        outputBuckets="buckets",
        **acts.examples.defaultKWArgs(
            allowSeparateRMax=seedingAlgorithmConfigArg.allowSeparateRMax,
            zBinNeighborsTop=seedingAlgorithmConfigArg.zBinNeighborsTop,
            zBinNeighborsBottom=seedingAlgorithmConfigArg.zBinNeighborsBottom,
            numPhiNeighbors=seedingAlgorithmConfigArg.numPhiNeighbors,
        ),
        gridConfig=gridConfig,
        gridOptions=gridOptions,
        seedFilterConfig=seedFilterConfig,
        seedFinderConfig=seedFinderConfig,
        seedFinderOptions=seedFinderOptions,
        hashingConfig=hashingCfg,
        hashingTrainingConfig=hashingTrainingCfg,
    )
    s.addAlgorithm(seedingAlg)

    if saveFiles:
        s.addWriter(
            acts.examples.CsvBucketWriter(
                level=customLogLevel(),
                inputBuckets=seedingAlg.config.outputBuckets,
                outputDir=str(outputDirRoot),
            )
        )
else:
    logger.fatal("unknown seedingAlgorithm %s", config.seedingAlgorithm)
    exit(f"unknown seedingAlgorithm {config.seedingAlgorithm}")

seeds = seedingAlg.config.outputSeeds

initialSigmas: Optional[list] = [
    1 * u.mm,
    1 * u.mm,
    1 * u.degree,
    1 * u.degree,
    0.1 / u.GeV,
    1 * u.ns,
]

initialVarInflation: Optional[list] = [1.0] * 6
particleHypothesis: Optional[acts.ParticleHypothesis] = acts.ParticleHypothesis.pion

parEstimateAlg = acts.examples.TrackParamsEstimationAlgorithm(
    level=logLevel,
    inputSeeds=seeds,
    outputTrackParameters="estimatedparameters",
    trackingGeometry=trackingGeometry,
    magneticField=field,
    **acts.examples.defaultKWArgs(
        initialSigmas=initialSigmas,
        initialVarInflation=initialVarInflation,
        particleHypothesis=particleHypothesis,
    ),
)
s.addAlgorithm(parEstimateAlg)

prototracks = "seed-prototracks"
s.addAlgorithm(
    acts.examples.SeedsToPrototracks(
        level=logLevel,
        inputSeeds=seeds,
        outputProtoTracks=prototracks,
    )
)

if outputDirRoot is not None:
    addSeedPerformanceWriters(
        s,
        outputDirRoot,
        seeds,
        prototracks,
        selectedParticles,
        inputParticles,
        parEstimateAlg.config.outputTrackParameters,
        logLevel,
    )

if outputDirRoot is not None:
    outputDirRoot = Path(outputDirRoot)
    if not outputDirRoot.exists():
        outputDirRoot.mkdir()
    # # s.addWriter(
    # #     acts.examples.TrackFinderPerformanceWriter(
    # #         level=customLogLevel(),
    # #         inputProtoTracks=inputProtoTracks,
    # #         inputParticles=selectedParticles,  # the original selected particles after digitization
    # #         inputMeasurementParticlesMap="measurement_particles_map",
    # #         filePath=str(outputDirRoot / "performance_seeding_trees.root"),
    # #     )
    # # )

    # s.addWriter(
    #     acts.examples.SeedingPerformanceWriter(
    #         level=customLogLevel(minLevel=acts.logging.DEBUG),
    #         inputProtoTracks=inputProtoTracks,
    #         inputParticles=selectedParticles,
    #         inputMeasurementParticlesMap="measurement_particles_map",
    #         filePath=str(outputDirRoot / "performance_seeding_hists.root"),
    #     )
    # )

    # # s.addWriter(
    # #     acts.examples.RootTrackParameterWriter(
    # #         level=customLogLevel(),
    # #         inputTrackParameters=parEstimateAlg.config.outputTrackParameters,
    # #         inputProtoTracks=parEstimateAlg.config.outputProtoTracks,
    # #         inputParticles=inputParticles,
    # #         inputSimHits="simhits",
    # #         inputMeasurementParticlesMap="measurement_particles_map",
    # #         inputMeasurementSimHitsMap="measurement_simhits_map",
    # #         filePath=str(outputDirRoot / "estimatedparams.root"),
    # #         treeName="estimatedparams",
    # #     )
    # # )

    # # if saveFiles:
    # #     s.addWriter(
    # #         acts.examples.CsvSimSeedWriter(
    # #             level=customLogLevel(),
    # #             inputSeeds=inputSeeds,
    # #             outputDir=str(outputDirRoot),
    # #         )
    # #     )

    s.addWriter(
        acts.examples.RootSeedWriter(
            level=customLogLevel(),
            inputSeeds=seeds,
            filePath=str(outputDirRoot / "seeds.root"),
        )
    )

# addCKFTracks(
#     s,
#     trackingGeometry,
#     field,
#     CKFPerformanceConfig(ptMin=1.0 * u.GeV, nMeasurementsMin=6),
#     TrackSelectorRanges(pt=(1.0 * u.GeV, None), absEta=(None, eta), removeNeutral=True),
#     outputDirRoot=outputDir,
#     # outputDirCsv=outputDir if saveFiles else None,
#     writeTrajectories=False,
# )

addCKFTracks(
    s,
    trackingGeometry,
    field,
    TrackSelectorConfig(
        pt=(1.0 * u.GeV, None),
        absEta=(None, eta),
        # loc0=(-4.0 * u.mm, 4.0 * u.mm),
        nMeasurementsMin=6,
    ),
    outputDirRoot=outputDir,
    # writeCovMat=True,
    writeTrajectories=False,
    twoWay=False,
    # outputDirCsv=outputDir,
)

# write track summary from CKF
# trackSummaryWriter = acts.examples.RootTrajectorySummaryWriter(
#     level=customLogLevel(),
#     inputTrajectories="ckfTrajectories",
#     # @note The full particles collection is used here to avoid lots of warnings
#     # since the unselected CKF track might have a majority particle not in the
#     # filtered particle collection. This could be avoided when a separate track
#     # selection algorithm is used.
#     inputParticles="particles_selected",
#     inputMeasurementParticlesMap="measurement_particles_map",
#     filePath=str(outputDirRoot / "tracksummary_ckf.root"),
#     treeName="tracksummary",
# )
# s.addWriter(trackSummaryWriter)

# addAmbiguityResolution(
#     s,
#     AmbiguityResolutionConfig(maximumSharedHits=3),
#     CKFPerformanceConfig(ptMin=1.0 * u.GeV, nMeasurementsMin=6),
#     outputDirRoot=outputDir,
# )

# addVertexFitting(
#     s,
#     field,
#     TrackSelectorRanges(pt=(1.0 * u.GeV, None), absEta=(None, 3.0), removeNeutral=True),
#     vertexFinder=VertexFinder.Iterative,
#     outputDirRoot=outputDir,
#     trajectories="trajectories",
# )

# if ambiguity_MLSolver:
#     addAmbiguityResolutionML(
#         s,
#         AmbiguityResolutionMLConfig(
#             maximumSharedHits=3, maximumIterations=1000000, nMeasurementsMin=7
#         ),
#         outputDirRoot=outputDir,
#         # outputDirCsv=outputDir,
#         onnxModelFile=os.path.dirname(__file__)
#         + "/MLAmbiguityResolution/duplicateClassifier.onnx",
#     )
# else:
#     addAmbiguityResolution(
#         s,
#         AmbiguityResolutionConfig(
#             maximumSharedHits=3, maximumIterations=1000000, nMeasurementsMin=7
#         ),
#         outputDirRoot=outputDir,
#         writeCovMat=True,
#         # outputDirCsv=outputDir,
#     )

# addVertexFitting(
#     s,
#     field,
#     vertexFinder=VertexFinder.Iterative,
#     outputDirRoot=outputDir,
# )

s.run()
