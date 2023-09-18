#!/usr/bin/env python3
import pathlib, acts, acts.examples
from acts.examples.simulation import (
    addParticleGun,
    MomentumConfig,
    EtaConfig,
    ParticleConfig,
    addPythia8,
    addFatras,
    ParticleSelectorConfig,
    addDigitization,
)
from acts.examples.reconstruction import (
    addSeeding,
    TruthSeedRanges,
    addCKFTracks,
    CKFPerformanceConfig,
    TrackSelectorRanges,
    addAmbiguityResolution,
    AmbiguityResolutionConfig,
    addVertexFitting,
    VertexFinder,
    SeedFinderOptionsArg,
)

import os
from collections import namedtuple
from pathlib import Path

from typing import Optional, Union
from enum import Enum

import argparse

eta = 4
# eta = 2.5

SeedingAlgorithm = Enum(
    "SeedingAlgorithm", "Default TruthSmeared TruthEstimated Orthogonal HashingSeeding"
)

DetectorName = Enum(
    "DetectorName", "ODD generic"
)

SeedFinderConfigName = Enum(
    "SeedFinderConfigName", "TrackML cpp"
)

HashingMetric = Enum(
    "HashingMetric", "dphi dR"
)

parser = argparse.ArgumentParser()
parser.add_argument("--mu",
                    type=int,
                    default=0)
parser.add_argument("--bucketSize",
                    type=int,
                    default=0)
parser.add_argument("--nevents",
                    type=int,
                    # default=1000)
                    default=100)
parser.add_argument("--maxSeedsPerSpM",
                    type=int,
                    default=1)
parser.add_argument("--seedingAlgorithm",
                    type=str)
parser.add_argument("--saveFilesSmall",
                    type=bool,
                    default=False)
parser.add_argument("--saveFiles",
                    type=bool,
                    default=False)
parser.add_argument("--AnnoySeed",
                    type=int,
                    default=123456789)
parser.add_argument("--zBins",
                    type=int,
                    default=0)
parser.add_argument("--phiBins",
                    type=int,
                    default=0)
parser.add_argument("--metric",
                    type=str)
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

print(mu, bucketSize, seedingAlgorithm)

def extractEnumName(enumvar):
    return str(enumvar).split(".")[-1]

u = acts.UnitConstants

def getActsExamplesDirectory():
    return Path(__file__).parent.parent.parent


Config = namedtuple('Config', ['mu', 'bucketSize', 'maxSeedsPerSpM', 'seedFinderConfig', 'detector', 'seedingAlgorithm', "metric", "AnnoySeed", "zBins", "phiBins"], 
defaults = [None, 0, 1, SeedFinderConfigName.TrackML, DetectorName.generic, SeedingAlgorithm.HashingSeeding, "angular", 123456789, 0, 0])
#https://stackoverflow.com/questions/34269772/type-hints-in-namedtuple
Config.__annotations__ = {'mu': int, 'bucketSize': int, 'maxSeedsPerSpM': int, 'seedFinderConfig': SeedFinderConfigName, 
'detector': DetectorName, 'seedingAlgorithm': SeedingAlgorithm, "metric": str, "AnnoySeed": int, "zBins": int, "phiBins": int}

# config = Config(mu=50, bucketSize=0, maxSeedsPerSpM=5, seedFinderConfig="cpp", detector=DetectorName.ODD)
# config = Config(mu=mu, bucketSize=bucketSize, maxSeedsPerSpM=1, seedFinderConfig="TrackML", detector=DetectorName.generic, seedingAlgorithm=SeedingAlgorithm.HashingSeeding)
config = Config(mu=mu, bucketSize=bucketSize, maxSeedsPerSpM=maxSeedsPerSpM, seedFinderConfig="TrackML", detector=DetectorName.generic, 
                seedingAlgorithm=seedingAlgorithm, metric=metric, AnnoySeed=AnnoySeed, zBins=zBins, phiBins=phiBins)
# config = Config(mu=mu, bucketSize=bucketSize, maxSeedsPerSpM=1, seedFinderConfig="TrackML", detector=DetectorName.ODD, 
#                 seedingAlgorithm=seedingAlgorithm, metric=metric, AnnoySeed=AnnoySeed, zBins=zBins)
# config = Config(mu=mu, bucketSize=bucketSize, maxSeedsPerSpM=1, seedFinderConfig="TrackML", detector=DetectorName.generic, seedingAlgorithm=seedingAlgorithm)

# config = Config(mu=mu, bucketSize=bucketSize, maxSeedsPerSpM=1000, seedFinderConfig="TrackML", detector=DetectorName.generic, 
#                 seedingAlgorithm=seedingAlgorithm, metric=metric, AnnoySeed=AnnoySeed, zBins=zBins, phiBins=phiBins)

actsExamplesDir = getActsExamplesDirectory()

if config.detector == DetectorName.ODD:
    from common import getOpenDataDetectorDirectory
    from acts.examples.odd import getOpenDataDetector

    geoDir = getOpenDataDetectorDirectory()

    oddMaterialMap = geoDir / "data/odd-material-maps.root"
    oddDigiConfig = geoDir / "config/odd-digi-smearing-config.json"
    oddSeedingSel = geoDir / "config/odd-seeding-config.json"
    oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)

    detector, trackingGeometry, decorators = getOpenDataDetector(
        geoDir, mdecorator=oddMaterialDeco
    )

    digiConfig = oddDigiConfig

    geoSelectionConfigFile = actsExamplesDir / "Algorithms/TrackFinding/share/geoSelection-openDataDetector.json"

elif config.detector == DetectorName.generic:
    print("Create detector and tracking geometry")

    detector, trackingGeometry, _ = acts.examples.GenericDetector.create()
    digiConfig = actsExamplesDir / "Algorithms/Digitization/share/default-smearing-config-generic.json"
    geoSelectionConfigFile = actsExamplesDir / "Algorithms/TrackFinding/share/geoSelection-genericDetector.json"
else:
    exit("Detector not supported")

truthSeedRanges = TruthSeedRanges(pt=(1.0 * u.GeV, None), eta=(-eta, eta), nHits=(9, None))

CKFptMin = 1.0 * u.GeV


doHashing = config.bucketSize > 0
bucketSize = config.bucketSize
npileup = config.mu
maxSeedsPerSpM = config.maxSeedsPerSpM

def get_dir_config(config:Config):
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

field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)

# if config.bucketSize == 0:
# s = acts.examples.Sequencer(events=5000, numThreads=1, outputDir=str(outputDir))
# else:
s = acts.examples.Sequencer(events=nevents, numThreads=1, outputDir=str(outputDir), enableEventTiming=True)
# s = acts.examples.Sequencer(events=10, numThreads=1, outputDir=str(outputDir))
# s = acts.examples.Sequencer(events=2, numThreads=1, outputDir=str(outputDir))

addPythia8(
    s,
    hardProcess=["Top:qqbar2ttbar=on"],
    # npileup=0,
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

addFatras(
    s,
    trackingGeometry,
    field,
    ParticleSelectorConfig(eta=(-eta, eta), pt=(150 * u.MeV, None), removeNeutral=True),
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

ParticleSmearingSigmas = namedtuple(
    "ParticleSmearingSigmas",
    ["d0", "d0PtA", "d0PtB", "z0", "z0PtA", "z0PtB", "t0", "phi", "theta", "pRel"],
    defaults=[None] * 10,
)

SeedFinderConfigArg = namedtuple(
    "SeedFinderConfig",
    [
        "maxSeedsPerSpM",
        "cotThetaMax",
        "sigmaScattering",
        "radLengthPerSeed",
        "minPt",
        "impactMax",
        "interactionPointCut",
        "arithmeticAverageCotTheta",
        "deltaZMax",
        "maxPtScattering",
        "zBinEdges",
        "skipPreviousTopSP",
        "zBinsCustomLooping",
        "rRangeMiddleSP",
        "useVariableMiddleSPRange",
        "binSizeR",
        "forceRadialSorting",
        "seedConfirmation",
        "centralSeedConfirmationRange",
        "forwardSeedConfirmationRange",
        "deltaR",  # (min,max)
        "deltaRBottomSP",  # (min,max)
        "deltaRTopSP",  # (min,max)
        "deltaRMiddleSPRange",  # (min,max)
        "collisionRegion",  # (min,max)
        "r",  # (min,max)
        "z",  # (min,max)
    ],
    defaults=[None] * 20 + [(None, None)] * 7,
)

SeedFinderOptionsArg = namedtuple(
    "SeedFinderOptions", ["beamPos", "bFieldInZ"], defaults=[(None, None), None]
    )

SeedFilterConfigArg = namedtuple(
    "SeedFilterConfig",
    [
        "impactWeightFactor",
        "zOriginWeightFactor",
        "compatSeedWeight",
        "compatSeedLimit",
        "numSeedIncrement",
        "seedWeightIncrement",
        "seedConfirmation",
        "curvatureSortingInFilter",
        "maxSeedsPerSpMConf",
        "maxQualitySeedsPerSpMConf",
        "useDeltaRorTopRadius",
        "deltaRMin",
    ],
    defaults=[None] * 12,
)

SpacePointGridConfigArg = namedtuple(
    "SeedGridConfig",
    [
        "rMax",
        "zBinEdges",
        "phiBinDeflectionCoverage",
        "impactMax",
        "deltaRMax",
        "phi",  # (min,max)
    ],
    defaults=[None] * 5 + [(None, None)] * 1,
)

SeedingAlgorithmConfigArg = namedtuple(
    "SeedingAlgorithmConfig",
    [
        "allowSeparateRMax",
        "zBinNeighborsTop",
        "zBinNeighborsBottom",
        "numPhiNeighbors",
    ],
    defaults=[None] * 4,
)

TrackParamsEstimationConfig = namedtuple(
    "TrackParamsEstimationConfig",
    [
        "deltaR",  # (min,max)
    ],
    defaults=[(None, None)],
)

seedingAlgorithm: SeedingAlgorithm = SeedingAlgorithm.Default
particleSmearingSigmas: ParticleSmearingSigmas = ParticleSmearingSigmas()
initialVarInflation: Optional[list] = None

import numpy as np
cotThetaMax = 1/(np.tan(2*np.arctan(np.exp(-eta))))# =1/tan(2×atan(e^(-eta)))
if config.seedFinderConfig == "TrackML":
    # seedFinderConfigArg = SeedfinderConfigArg(
    #         r=(None, 200 * u.mm),  # rMin=default, 33mm
    #         deltaR=(1 * u.mm, 60 * u.mm),
    #         collisionRegion=(-250 * u.mm, 250 * u.mm),
    #         z=(-2000 * u.mm, 2000 * u.mm),
    #         maxSeedsPerSpM=maxSeedsPerSpM,
    #         sigmaScattering=5,
    #         radLengthPerSeed=0.1,
    #         minPt=500 * u.MeV,
    #         bFieldInZ=1.99724 * u.T,
    #         impactMax=3 * u.mm,
    #         # cotThetaMax = 1000,
    #     )
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
            cotThetaMax=cotThetaMax # =1/tan(2×atan(e^(-eta)))
            # cotThetaMax = 1000, # Hashing better perfs with that; in SPGrid: float zBinSize = config.cotThetaMax * config.deltaRMax; 
            # int zBins = max(1, (int)std::floor((config.zMax - config.zMin) / zBinSize))
    )
elif config.seedFinderConfig == "cpp":
    seedFinderConfigArg = SeedFinderConfigArg(maxSeedsPerSpM=maxSeedsPerSpM, cotThetaMax=cotThetaMax)


seedFinderOptionsArg: SeedFinderOptionsArg = SeedFinderOptionsArg(bFieldInZ=1.99724 * u.T)
seedFilterConfigArg: SeedFilterConfigArg = SeedFilterConfigArg()
spacePointGridConfigArg: SpacePointGridConfigArg = SpacePointGridConfigArg()
seedingAlgorithmConfigArg: SeedingAlgorithmConfigArg = SeedingAlgorithmConfigArg()
trackParamsEstimationConfig: TrackParamsEstimationConfig = TrackParamsEstimationConfig()
inputParticles: str = "particles_initial"
outputDirRoot: Optional[Union[Path, str]] = outputDir
logLevel: Optional[acts.logging.Level] = None
rnd: Optional[acts.examples.RandomNumbers] = None

customLogLevel = acts.examples.defaultLogging(s, logLevel)
logger = acts.logging.getLogger("addSeeding")

if truthSeedRanges is not None:
    selAlg = acts.examples.TruthSeedSelector(
        **acts.examples.defaultKWArgs(
            ptMin=truthSeedRanges.pt[0],
            ptMax=truthSeedRanges.pt[1],
            etaMin=truthSeedRanges.eta[0],
            etaMax=truthSeedRanges.eta[1],
            nHitsMin=truthSeedRanges.nHits[0],
            nHitsMax=truthSeedRanges.nHits[1],
            rhoMin=truthSeedRanges.rho[0],
            rhoMax=truthSeedRanges.rho[1],
            zMin=truthSeedRanges.z[0],
            zMax=truthSeedRanges.z[1],
            phiMin=truthSeedRanges.phi[0],
            phiMax=truthSeedRanges.phi[1],
            absEtaMin=truthSeedRanges.absEta[0],
            absEtaMax=truthSeedRanges.absEta[1],
        ),
        level=acts.logging.INFO,
        inputParticles=inputParticles,
        inputMeasurementParticlesMap="measurement_particles_map",
        outputParticles="truth_seeds_selected",
    )
    s.addAlgorithm(selAlg)
    selectedParticles = selAlg.config.outputParticles
else:
    selectedParticles = inputParticles

print(selectedParticles)

# Create starting parameters from either particle smearing or combined seed
# finding and track parameters estimation
spAlg = acts.examples.SpacePointMaker(
    level=acts.logging.INFO,
    inputSourceLinks="sourcelinks",
    inputMeasurements="measurements",
    outputSpacePoints="spacepoints",
    trackingGeometry=trackingGeometry,
    geometrySelection=acts.examples.readJsonGeometryList(
        str(geoSelectionConfigFile)
    ),
)
s.addAlgorithm(spAlg)

# if saveFiles:
#     s.addWriter(
#         acts.examples.CsvSpacepointWriter(
#             level=customLogLevel(),
#             inputSpacepoints="spacepoints",
#             outputDir=str(outputDirRoot),
#         )
#     )

outputDirRoot = Path(outputDirRoot)
if not outputDirRoot.exists():
    outputDirRoot.mkdir()
s.addWriter(
    acts.examples.RootSpacepointWriter(
        level=customLogLevel(),
        inputSpacepoints="spacepoints",
        filePath=str(outputDirRoot / "spacepoints.root")
    )
)

# s.config.logLevel = acts.logging.VERBOSE

from typing import Optional, Union, List

def addHashing(
    s: acts.examples.Sequencer,
    bucketSize: Optional[int]=10,
    AnnoySeed: Optional[int]=123456789,
    zBins: Optional[int]=0,
    phiBins: Optional[int]=100,
    metric = HashingMetric.dphi,
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
    hashingTrainingCfg = acts.examples.HashingTrainingAlgorithm.Config(
        inputSpacePoints="spacepoints",
        AnnoySeed=AnnoySeed,
        f=f,
    )

    # addHashingTraining
    hashingTrainingAlg = acts.examples.HashingTrainingAlgorithm(hashingTrainingCfg, 
                                                                # acts.logging.VERBOSE
                                                                s.config.logLevel
                                                                )

    s.addAlgorithm(hashingTrainingAlg)

    hashingCfg = acts.examples.HashingAlgorithm.Config(
        inputSpacePoints="spacepoints",
        bucketSize=bucketSize,
        zBins=zBins,
        phiBins=phiBins,
    )

    hashingAlg = acts.examples.HashingAlgorithm(hashingCfg, 
                                                # acts.logging.VERBOSE
                                                s.config.logLevel
                                                )

    s.addAlgorithm(hashingAlg)

    return s

def mergeSeeds(
    s: acts.examples.Sequencer,
    inputSeedNames: List[str],
    inputProtoTrackNames: List[str],
) -> acts.examples.Sequencer:
    """This function steers the digitization step

    Parameters
    ----------
    s: Sequencer
        the sequencer module to which we add the Digitization steps (returned from addDigitization)
    inputSeedNames : seeds to merge
    inputProtoTrackNames : prototracks to merge
    outputDirCsv : Path|str, path, None
        the output folder for the Csv output, None triggers no output
    outputDirRoot : Path|str, path, None
        the output folder for the Root output, None triggers no output
    """

    if int(s.config.logLevel) <= int(acts.logging.DEBUG):
        acts.examples.dump_args_calls(locals())

    # Merging
    mergingCfg = acts.examples.MergeSeedsAlgorithm.Config(
        inputSeeds=inputSeedNames,
        inputProtoTracks=inputProtoTrackNames,
        outputSeeds="seeds",
        outputProtoTracks="prototracks",
    )
    mergingAlg = acts.examples.MergeSeedsAlgorithm(mergingCfg, s.config.logLevel)

    s.addAlgorithm(mergingAlg)

    return s

if doHashing:
    # for now hashing only use space points and not clusters
    print("***> addHashing")
    s = addHashing(
        s,
        bucketSize=bucketSize,
        AnnoySeed=AnnoySeed,
        metric=config.metric,
        zBins=zBins,
        phiBins=phiBins,
    )

    if saveFiles:
        s.addWriter(
            acts.examples.CsvBucketWriter(
                level=customLogLevel(),
                inputBuckets="buckets",
                outputDir=str(outputDirRoot),
            )
        )

# Use seeding
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
        maxSeedsPerSpM=seedFinderConfigArg.maxSeedsPerSpM,
        cotThetaMax=seedFinderConfigArg.cotThetaMax,
        sigmaScattering=seedFinderConfigArg.sigmaScattering,
        radLengthPerSeed=seedFinderConfigArg.radLengthPerSeed,
        minPt=seedFinderConfigArg.minPt,
        impactMax=seedFinderConfigArg.impactMax,
        interactionPointCut=seedFinderConfigArg.interactionPointCut,
        arithmeticAverageCotTheta=seedFinderConfigArg.arithmeticAverageCotTheta,
        deltaZMax=seedFinderConfigArg.deltaZMax,
        maxPtScattering=seedFinderConfigArg.maxPtScattering,
        zBinEdges=seedFinderConfigArg.zBinEdges,
        skipPreviousTopSP=seedFinderConfigArg.skipPreviousTopSP,
        zBinsCustomLooping=seedFinderConfigArg.zBinsCustomLooping,
        rRangeMiddleSP=seedFinderConfigArg.rRangeMiddleSP,
        useVariableMiddleSPRange=seedFinderConfigArg.useVariableMiddleSPRange,
        binSizeR=seedFinderConfigArg.binSizeR,
        forceRadialSorting=seedFinderConfigArg.forceRadialSorting,
        seedConfirmation=seedFinderConfigArg.seedConfirmation,
        centralSeedConfirmationRange=seedFinderConfigArg.centralSeedConfirmationRange,
        forwardSeedConfirmationRange=seedFinderConfigArg.forwardSeedConfirmationRange,
    ),
)

seedFinderOptions = acts.SeedFinderOptions(
    **acts.examples.defaultKWArgs(
        beamPos=acts.Vector2(0.0, 0.0)
        if seedFinderOptionsArg.beamPos == (None, None)
        else acts.Vector2(
            seedFinderOptionsArg.beamPos[0], seedFinderOptionsArg.beamPos[1]
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
        curvatureSortingInFilter=seedFilterConfigArg.curvatureSortingInFilter,
        maxSeedsPerSpMConf=seedFilterConfigArg.maxSeedsPerSpMConf,
        maxQualitySeedsPerSpMConf=seedFilterConfigArg.maxQualitySeedsPerSpMConf,
        useDeltaRorTopRadius=seedFilterConfigArg.useDeltaRorTopRadius,
    )
)

gridConfig = acts.SpacePointGridConfig(
    **acts.examples.defaultKWArgs(
        bFieldInZ=seedFinderOptions.bFieldInZ,
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
        impactMax=spacePointGridConfigArg.impactMax,
        zBinEdges=spacePointGridConfigArg.zBinEdges,
        phiBinDeflectionCoverage=spacePointGridConfigArg.phiBinDeflectionCoverage,
    )
)

if config.seedingAlgorithm == SeedingAlgorithm.Default:
    seedNames = []
    protoTrackNames = []
    nBucketsLimit = 1
    for bucketNumber in range(nBucketsLimit):
        if doHashing:
            bucketSuffix = "{}".format(bucketNumber)
            bucketSP = "hashingSPBucket_{}".format(bucketNumber)
        else:
            bucketSuffix = ""
            bucketSP = "spacepoints"
        logger.info("Using default seeding")

        seedingAlg = acts.examples.SeedingAlgorithm(
            level=customLogLevel(),
            # inputSpacePoints=[spAlg.config.outputSpacePoints],    
            inputSpacePoints=[bucketSP],    
            outputSeeds="seeds{}".format(bucketSuffix),
            outputProtoTracks="prototracks{}".format(bucketSuffix),
            **acts.examples.defaultKWArgs(
                allowSeparateRMax=seedingAlgorithmConfigArg.allowSeparateRMax,
                zBinNeighborsTop=seedingAlgorithmConfigArg.zBinNeighborsTop,
                zBinNeighborsBottom=seedingAlgorithmConfigArg.zBinNeighborsBottom,
                numPhiNeighbors=seedingAlgorithmConfigArg.numPhiNeighbors,
            ),
            gridConfig=gridConfig,
            seedFilterConfig=seedFilterConfig,
            seedFinderConfig=seedFinderConfig,
            seedFinderOptions=seedFinderOptions,
        )
        s.addAlgorithm(seedingAlg)
        seedNames.append(seedingAlg.config.outputSeeds)
        protoTrackNames.append(seedingAlg.config.outputProtoTracks)

    if doHashing:
        s = mergeSeeds(s, seedNames, protoTrackNames)
elif config.seedingAlgorithm == SeedingAlgorithm.HashingSeeding:
    # assert(doHashing)
    bucket_list = []
    # if doHashing:
    #     bucketSP = "hashingSPBucket_{}".format(0)
    # else:
    #     bucketSP = "spacepoints"
    bucketSP = "buckets"
    bucket_list.append(bucketSP)
    logger.info("Using Hashing seeding")

    seedingAlg = acts.examples.SeedingAlgorithmHashing(
        level=customLogLevel(),
        # inputSpacePoints=[spAlg.config.outputSpacePoints],    
        inputSpacePoints=bucket_list,    
        outputSeeds="seeds",
        outputProtoTracks="prototracks",
        **acts.examples.defaultKWArgs(
            allowSeparateRMax=seedingAlgorithmConfigArg.allowSeparateRMax,
            zBinNeighborsTop=seedingAlgorithmConfigArg.zBinNeighborsTop,
            zBinNeighborsBottom=seedingAlgorithmConfigArg.zBinNeighborsBottom,
            numPhiNeighbors=seedingAlgorithmConfigArg.numPhiNeighbors,
        ),
        gridConfig=gridConfig,
        seedFilterConfig=seedFilterConfig,
        seedFinderConfig=seedFinderConfig,
        seedFinderOptions=seedFinderOptions,
    )
    s.addAlgorithm(seedingAlg)
elif config.seedingAlgorithm == SeedingAlgorithm.Orthogonal:
    logger.info("Using orthogonal seeding")
    # Use seeding
    seedFinderConfig = acts.SeedFinderOrthogonalConfig(
        **acts.examples.defaultKWArgs(
            rMin=seedFinderConfigArg.r[0],
            rMax=seedFinderConfigArg.r[1],
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
            collisionRegionMin=seedFinderConfigArg.collisionRegion[0],
            collisionRegionMax=seedFinderConfigArg.collisionRegion[1],
            zMin=seedFinderConfigArg.z[0],
            zMax=seedFinderConfigArg.z[1],
            maxSeedsPerSpM=seedFinderConfigArg.maxSeedsPerSpM,
            cotThetaMax=seedFinderConfigArg.cotThetaMax,
            sigmaScattering=seedFinderConfigArg.sigmaScattering,
            radLengthPerSeed=seedFinderConfigArg.radLengthPerSeed,
            minPt=seedFinderConfigArg.minPt,
            impactMax=seedFinderConfigArg.impactMax,
            interactionPointCut=seedFinderConfigArg.interactionPointCut,
            deltaZMax=seedFinderConfigArg.deltaZMax,
            maxPtScattering=seedFinderConfigArg.maxPtScattering,
            rRangeMiddleSP=seedFinderConfigArg.rRangeMiddleSP,
            useVariableMiddleSPRange=seedFinderConfigArg.useVariableMiddleSPRange,
            seedConfirmation=seedFinderConfigArg.seedConfirmation,
            centralSeedConfirmationRange=seedFinderConfigArg.centralSeedConfirmationRange,
            forwardSeedConfirmationRange=seedFinderConfigArg.forwardSeedConfirmationRange,
        ),
    )

    
    # seedFinderOptions = SeedFinderOptionsArg(
    #     **acts.examples.defaultKWArgs(
    #         bFieldInZ=seedFinderOptionsArg.bFieldInZ,
    #         beamPos=acts.Vector2(0.0, 0.0)
    #         if seedFinderOptionsArg.beamPos == (None, None)
    #         else seedFinderOptionsArg.beamPos,
    #     )
    # )

    seedFilterConfig = acts.SeedFilterConfig(
        **acts.examples.defaultKWArgs(
            maxSeedsPerSpM=seedFinderConfig.maxSeedsPerSpM,
            deltaRMin=(
                seedFinderConfigArg.deltaR[0]
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
            curvatureSortingInFilter=seedFilterConfigArg.curvatureSortingInFilter,
            maxSeedsPerSpMConf=seedFilterConfigArg.maxSeedsPerSpMConf,
            maxQualitySeedsPerSpMConf=seedFilterConfigArg.maxQualitySeedsPerSpMConf,
            useDeltaRorTopRadius=seedFilterConfigArg.useDeltaRorTopRadius,
        )
    )

    seedingAlg = acts.examples.SeedingOrthogonalAlgorithm(
        level=customLogLevel(),
        inputSpacePoints=[spAlg.config.outputSpacePoints],
        outputSeeds="seeds",
        outputProtoTracks="prototracks",
        seedFilterConfig=seedFilterConfig,
        seedFinderConfig=seedFinderConfig,
        seedFinderOptions=seedFinderOptions,
    )
    s.addAlgorithm(seedingAlg)
    inputProtoTracks = seedingAlg.config.outputProtoTracks
    inputSeeds = seedingAlg.config.outputSeeds
else:
    logger.fatal("unknown seedingAlgorithm %s", seedingAlgorithm)

# inputProtoTracks = seedingAlg.config.outputProtoTracks
# inputSeeds = seedingAlg.config.outputSeeds

inputProtoTracks = "prototracks"#mergingCfg.config.outputProtoTracks
inputSeeds = "seeds"#mergingCfg.config.outputSeeds

parEstimateAlg = acts.examples.TrackParamsEstimationAlgorithm(
    level=acts.logging.INFO,
    inputSeeds=inputSeeds,
    inputProtoTracks=inputProtoTracks,
    inputSpacePoints=[spAlg.config.outputSpacePoints],
    inputSourceLinks=spAlg.config.inputSourceLinks,
    outputTrackParameters="estimatedparameters",
    outputProtoTracks="prototracks_estimated",
    trackingGeometry=trackingGeometry,
    magneticField=field,
    **acts.examples.defaultKWArgs(
        initialVarInflation=initialVarInflation,
        deltaRMin=trackParamsEstimationConfig.deltaR[0],
        deltaRMax=trackParamsEstimationConfig.deltaR[1],
    ),
)
s.addAlgorithm(parEstimateAlg)

if outputDirRoot is not None:
    outputDirRoot = Path(outputDirRoot)
    if not outputDirRoot.exists():
        outputDirRoot.mkdir()
    # s.addWriter(
    #     acts.examples.TrackFinderPerformanceWriter(
    #         level=customLogLevel(),
    #         inputProtoTracks=inputProtoTracks,
    #         inputParticles=selectedParticles,  # the original selected particles after digitization
    #         inputMeasurementParticlesMap="measurement_particles_map",
    #         filePath=str(outputDirRoot / "performance_seeding_trees.root"),
    #     )
    # )

    s.addWriter(
        acts.examples.SeedingPerformanceWriter(
            level=customLogLevel(minLevel=acts.logging.DEBUG),
            inputProtoTracks=inputProtoTracks,
            inputParticles=selectedParticles,
            inputMeasurementParticlesMap="measurement_particles_map",
            filePath=str(outputDirRoot / "performance_seeding_hists.root"),
        )
    )

    # s.addWriter(
    #     acts.examples.RootTrackParameterWriter(
    #         level=customLogLevel(),
    #         inputTrackParameters=parEstimateAlg.config.outputTrackParameters,
    #         inputProtoTracks=parEstimateAlg.config.outputProtoTracks,
    #         inputParticles=inputParticles,
    #         inputSimHits="simhits",
    #         inputMeasurementParticlesMap="measurement_particles_map",
    #         inputMeasurementSimHitsMap="measurement_simhits_map",
    #         filePath=str(outputDirRoot / "estimatedparams.root"),
    #         treeName="estimatedparams",
    #     )
    # )

    # if saveFiles:
    #     s.addWriter(
    #         acts.examples.CsvSimSeedWriter(
    #             level=customLogLevel(),
    #             inputSeeds=inputSeeds,
    #             outputDir=str(outputDirRoot),
    #         )
    #     )

    s.addWriter(
        acts.examples.RootSeedWriter(
            level=customLogLevel(),
            inputSeeds=inputSeeds,
            filePath=str(outputDirRoot / "seeds.root")
        )
    )

addCKFTracks(
    s,
    trackingGeometry,
    field,
    CKFPerformanceConfig(ptMin=1.0 * u.GeV, nMeasurementsMin=6),
    TrackSelectorRanges(pt=(1.0 * u.GeV, None), absEta=(None, eta), removeNeutral=True),
    outputDirRoot=outputDir,
    # outputDirCsv=outputDir if saveFiles else None,
    writeTrajectories=False,
)

# write track summary from CKF
# trackSummaryWriter = acts.examples.RootTrajectorySummaryWriter(
#     level=customLogLevel(),
#     inputTrajectories="ckfTrajectories",
#     # @note The full particles collection is used here to avoid lots of warnings
#     # since the unselected CKF track might have a majority particle not in the
#     # filtered particle collection. This could be avoided when a seperate track
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

s.run()
