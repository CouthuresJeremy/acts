#!/usr/bin/env python3
import pathlib, acts, acts.examples, acts.examples.itk
from acts.examples.simulation import (
    addParticleGun,
    MomentumConfig,
    EtaConfig,
    ParticleConfig,
    addPythia8,
    addFatras,
    ParticleSelectorConfig,
    addDigitization,
    addGeant4,
)
from acts.examples.reconstruction import (
    addSeeding,
    SeedingAlgorithm,
    TruthSeedRanges,
    addCKFTracks,
    TrackSelectorConfig,
    addAmbiguityResolution,
    AmbiguityResolutionConfig,
    addVertexFitting,
    VertexFinder,
)

g4_simulation = True
ttbar_pu200 = True
u = acts.UnitConstants
geo_dir = pathlib.Path("acts-itk")
outputDir = pathlib.Path.cwd() / "itk_output"
# acts.examples.dump_args_calls(locals())  # show acts.examples python binding calls

detector, trackingGeometry, decorators = acts.examples.itk.buildITkGeometry(geo_dir)
field = acts.examples.MagneticFieldMapXyz(str(geo_dir / "bfield/ATLAS-BField-xyz.root"))
rnd = acts.examples.RandomNumbers(seed=42)

from acts.examples.dd4hep import DD4hepDetector
from acts.examples.geant4.dd4hep import DDG4DetectorConstructionFactory

if type(detector) is DD4hepDetector:
    print("Using DD4hep detector")
else:
    print("Using ACTS detector")

from acts.examples import TelescopeDetector
from acts.examples.geant4 import TelescopeG4DetectorConstructionFactory

if type(detector) is TelescopeDetector:
    TelescopeG4DetectorConstructionFactory(detector)

s = acts.examples.Sequencer(events=100, numThreads=1, outputDir=str(outputDir), trackFpes=False, enableEventTiming=True)

if not ttbar_pu200:
    addParticleGun(
        s,
        MomentumConfig(1.0 * u.GeV, 10.0 * u.GeV, transverse=True),
        EtaConfig(-4.0, 4.0, uniform=True),
        ParticleConfig(2, acts.PdgParticle.eMuon, randomizeCharge=True),
        rnd=rnd,
    )
else:
    addPythia8(
        s,
        hardProcess=["Top:qqbar2ttbar=on"],
        npileup=50,
        vtxGen=acts.examples.GaussianVertexGenerator(
            stddev=acts.Vector4(0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns),
            mean=acts.Vector4(0, 0, 0, 0),
        ),
        rnd=rnd,
        outputDirRoot=outputDir,
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
        rnd=rnd,
        preSelectParticles=ParticleSelectorConfig(
            rho=(0.0 * u.mm, 28.0 * u.mm),
            absZ=(0.0 * u.mm, 1.0 * u.m),
            eta=(-4.0, 4.0),
            pt=(150 * u.MeV, None),
            removeNeutral=True,
        )
        if ttbar_pu200
        else ParticleSelectorConfig(),
        outputDirRoot=outputDir,
    )

addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=geo_dir / "itk-hgtd/itk-smearing-config.json",
    outputDirRoot=outputDir,
    rnd=rnd,
)

addSeeding(
    s,
    trackingGeometry,
    field,
    TruthSeedRanges(pt=(1.0 * u.GeV, None), eta=(-4.0, 4.0), nHits=(9, None))
    if ttbar_pu200
    else TruthSeedRanges(),
    seedingAlgorithm=SeedingAlgorithm.Default,
    *acts.examples.itk.itkSeedingAlgConfig(
        acts.examples.itk.InputSpacePointsType.PixelSpacePoints
    ),
    initialSigmas=[
        1 * u.mm,
        1 * u.mm,
        1 * u.degree,
        1 * u.degree,
        0.1 / u.GeV,
        1 * u.ns,
    ],
    initialVarInflation=[1.0] * 6,
    geoSelectionConfigFile=geo_dir / "itk-hgtd/geoSelection-ITk.json",
    outputDirRoot=outputDir,
)

s.addWriter(
    acts.examples.RootSpacepointWriter(
        level=acts.logging.INFO,
        inputSpacepoints="spacepoints",
        filePath=str(outputDir / "spacepoints.root")
    )
)

addCKFTracks(
    s,
    trackingGeometry,
    field,
    trackSelectorConfig=(
        # fmt: off
        TrackSelectorConfig(absEta=(None, 2.0), pt=(0.9 * u.GeV, None), nMeasurementsMin=9, maxHoles=2, maxSharedHits=2),
        TrackSelectorConfig(absEta=(None, 2.6), pt=(0.4 * u.GeV, None), nMeasurementsMin=8, maxHoles=2, maxSharedHits=2),
        TrackSelectorConfig(absEta=(None, 4.0), pt=(0.4 * u.GeV, None), nMeasurementsMin=7, maxHoles=2, maxSharedHits=2),
        # fmt: on
    ),
    outputDirRoot=outputDir,
)

addAmbiguityResolution(
    s,
    AmbiguityResolutionConfig(
        maximumSharedHits=3,
        maximumIterations=10000,
        nMeasurementsMin=6,
    ),
    outputDirRoot=outputDir,
)

addVertexFitting(
    s,
    field,
    vertexFinder=VertexFinder.Iterative,
    outputDirRoot=outputDir,
)

s.run()
