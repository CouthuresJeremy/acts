#!/usr/bin/env python3
import os, argparse, pathlib, acts, acts.examples

from pathlib import Path

from typing import Optional, Union
from enum import Enum

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
    addSeedingTruthSelection,
    addSpacePointsMaking,
)
from common import getOpenDataDetectorDirectory
from acts.examples.odd import getOpenDataDetector

parser = argparse.ArgumentParser(description="Full chain with the OpenDataDetector")

parser.add_argument("--events", "-n", help="Number of events", type=int, default=100)
parser.add_argument(
    "--geant4", help="Use Geant4 instead of fatras", action="store_true"
)
parser.add_argument(
    "--ttbar",
    help="Use Pythia8 (ttbar, pile-up 200) instead of particle gun",
    action="store_true",
)
parser.add_argument(
    "--MLSolver",
    help="Use the Ml Ambiguity Solver instead of the classical one",
    action="store_true",
)

args = vars(parser.parse_args())

eta = 4

ttbar = args["ttbar"]
g4_simulation = args["geant4"]
ambiguity_MLSolver = args["MLSolver"]
u = acts.UnitConstants
geoDir = getOpenDataDetectorDirectory()
outputDir = pathlib.Path.cwd() / "test_acts" / "odd_output"
# acts.examples.dump_args_calls(locals())  # show python binding calls

oddMaterialMap = geoDir / "data/odd-material-maps.root"
oddDigiConfig = geoDir / "config/odd-digi-smearing-config.json"
oddSeedingSel = geoDir / "config/odd-seeding-config.json"
oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)

detector, trackingGeometry, decorators = getOpenDataDetector(
    geoDir, mdecorator=oddMaterialDeco
)

geoSelectionConfigFile = oddSeedingSel


field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)

s = acts.examples.Sequencer(
    events=args["events"],
    numThreads=1,
    outputDir=str(outputDir),
)

if not ttbar:
    addParticleGun(
        s,
        MomentumConfig(1.0 * u.GeV, 10.0 * u.GeV, transverse=True),
        EtaConfig(-eta, eta),
        PhiConfig(0.0, 360.0 * u.degree),
        ParticleConfig(4, acts.PdgParticle.eMuon, randomizeCharge=True),
        vtxGen=acts.examples.GaussianVertexGenerator(
            mean=acts.Vector4(0, 0, 0, 0),
            stddev=acts.Vector4(0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 1.0 * u.ns),
        ),
        multiplicity=200,
        rnd=rnd,
    )
else:
    addPythia8(
        s,
        hardProcess=["Top:qqbar2ttbar=on"],
        npileup=50,
        vtxGen=acts.examples.GaussianVertexGenerator(
            mean=acts.Vector4(0, 0, 0, 0),
            stddev=acts.Vector4(0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns),
        ),
        rnd=rnd,
        outputDirRoot=outputDir,
        # outputDirCsv=outputDir,
    )
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
        preSelectParticles=ParticleSelectorConfig(
            rho=(0.0, 24 * u.mm),
            absZ=(0.0, 1.0 * u.m),
            eta=(-eta, eta),
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
        preSelectParticles=(
            ParticleSelectorConfig(
                rho=(0.0, 24 * u.mm),
                absZ=(0.0, 1.0 * u.m),
                eta=(-eta, eta),
                pt=(150 * u.MeV, None),
                removeNeutral=True,
            )
            if ttbar
            else ParticleSelectorConfig()
        ),
        enableInteractions=False,
        outputDirRoot=outputDir,
        # outputDirCsv=outputDir,
        rnd=rnd,
    )

addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=oddDigiConfig,
    outputDirRoot=outputDir,
    # outputDirCsv=outputDir,
    rnd=rnd,
)

import acts.examples.reconstruction as reconstruction

SeedFinderConfigArg = reconstruction.SeedFinderConfigArg
SeedFinderOptionsArg = reconstruction.SeedFinderOptionsArg
SeedFilterConfigArg = reconstruction.SeedFilterConfigArg
SpacePointGridConfigArg = reconstruction.SpacePointGridConfigArg
SeedingAlgorithmConfigArg = reconstruction.SeedingAlgorithmConfigArg


initialVarInflation = None

import numpy as np

cotThetaMax = 1 / (np.tan(2 * np.arctan(np.exp(-eta))))  # =1/tan(2×atan(e^(-eta)))
maxSeedsPerSpM = 1

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

seedFinderOptionsArg: SeedFinderOptionsArg = SeedFinderOptionsArg(
    bFieldInZ=1.99724 * u.T
)
seedFilterConfigArg: SeedFilterConfigArg = SeedFilterConfigArg()
spacePointGridConfigArg: SpacePointGridConfigArg = SpacePointGridConfigArg()
seedingAlgorithmConfigArg: SeedingAlgorithmConfigArg = SeedingAlgorithmConfigArg()
inputParticles: str = "particles"
outputDirRoot = outputDir
logLevel = None

truthSeedRanges = (
    TruthSeedRanges(pt=(1.0 * u.GeV, None), eta=(-eta, eta), nHits=(9, None))
    if ttbar
    else TruthSeedRanges()
)

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
        # inputSpacepoints="spacepoints",
        inputSpacepoints=spacePoints,
        filePath=str(outputDirRoot / "spacepoints.root"),
    )
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
        skipZMiddleBinSearch=seedFinderConfigArg.skipZMiddleBinSearch,
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

logger.info("Using default seeding")
logLevel = acts.examples.defaultLogging(s, logLevel)()

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


seeds = seedingAlg.config.outputSeeds

initialSigmas: Optional[list] = None
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

# if outputDirRoot is not None:
#     addSeedPerformanceWriters(
#         s,
#         outputDirRoot,
#         seeds,
#         prototracks,
#         selectedParticles,
#         inputParticles,
#         parEstimateAlg.config.outputTrackParameters,
#         logLevel,
#     )

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

# addSeeding(
#     s,
#     trackingGeometry,
#     field,
#     TruthSeedRanges(pt=(1.0 * u.GeV, None), eta=(-eta, eta), nHits=(9, None))
#     if ttbar
#     else TruthSeedRanges(),
#     geoSelectionConfigFile=oddSeedingSel,
#     outputDirRoot=outputDir,
# )

addCKFTracks(
    s,
    trackingGeometry,
    field,
    TrackSelectorConfig(
        pt=(1.0 * u.GeV if ttbar else 0.0, None),
        absEta=(None, eta),
        loc0=(-4.0 * u.mm, 4.0 * u.mm),
        nMeasurementsMin=7,
    ),
    outputDirRoot=outputDir,
    writeCovMat=True,
    # outputDirCsv=outputDir,
)

if ambiguity_MLSolver:
    addAmbiguityResolutionML(
        s,
        AmbiguityResolutionMLConfig(
            maximumSharedHits=3, maximumIterations=1000000, nMeasurementsMin=7
        ),
        outputDirRoot=outputDir,
        # outputDirCsv=outputDir,
        onnxModelFile=os.path.dirname(__file__)
        + "/MLAmbiguityResolution/duplicateClassifier.onnx",
    )
else:
    addAmbiguityResolution(
        s,
        AmbiguityResolutionConfig(
            maximumSharedHits=3, maximumIterations=1000000, nMeasurementsMin=7
        ),
        outputDirRoot=outputDir,
        writeCovMat=True,
        # outputDirCsv=outputDir,
    )

addVertexFitting(
    s,
    field,
    vertexFinder=VertexFinder.Iterative,
    outputDirRoot=outputDir,
)

s.run()
