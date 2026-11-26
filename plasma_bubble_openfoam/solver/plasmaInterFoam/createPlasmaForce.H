// ---------------------------------------------------------------------------
// Phenomenological plasma (EHD) forcing for a discharge bubble at a needle.
//   1) mean EHD force on the gas, central and attracting toward the discharge
//      point x0:  f = -F0 * S(d) * tanh(d/d0) * (1 - alpha1) * d_hat,
//      S(d) = 0.5*(1 - tanh((d - RE)/wE)).  Curl-free in each phase.
//   2) fluctuating normal interfacial stress p'(theta,t) * grad(alpha1),
//      p' = Pfl * Sfl(d) * xi(theta,t), xi a sum of Ornstein-Uhlenbeck
//      driven angular modes (theta = polar angle about x0). Its amplitude is
//      what distinguishes the two polarities (Wang et al., Matter 2024, Fig. 4).
// Read from constant/plasmaProperties.
// ---------------------------------------------------------------------------
IOdictionary plasmaDict
(
    IOobject
    (
        "plasmaProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);

const scalar plF0  = plasmaDict.get<scalar>("F0");     // N/m^3
const scalar plRE  = plasmaDict.get<scalar>("RE");     // m
const scalar plwE  = plasmaDict.get<scalar>("wE");     // m
const scalar pld0  = plasmaDict.get<scalar>("d0");     // m
const vector plX0  = plasmaDict.get<vector>("x0");     // m
const vector plAx  = plasmaDict.get<vector>("axis");   // unit
const scalar plPfl = plasmaDict.get<scalar>("Pfl");    // Pa
const scalar plRfl = plasmaDict.get<scalar>("Rfl");    // m
const scalar plTau = plasmaDict.get<scalar>("tau");    // s
const label  plM0  = plasmaDict.get<label>("mMin");
const label  plM1  = plasmaDict.get<label>("mMax");
const label  plSeed = plasmaDict.get<label>("seed");
// mean outward (ion-wind) stress on the upper interface: Pup*max(cos theta,0)^nUp
const scalar plPup = plasmaDict.getOrDefault<scalar>("Pup", 0);   // Pa
const scalar plnUp = plasmaDict.getOrDefault<scalar>("nUp", 2);

Random plRnd(plSeed);
const label plNm = plM1 - plM0 + 1;
scalarField plA(plNm, Zero), plB(plNm, Zero);
forAll(plA, k)
{
    plA[k] = plRnd.GaussNormal<scalar>();
    plB[k] = plRnd.GaussNormal<scalar>();
}

surfaceScalarField plasmaForcef
(
    IOobject("plasmaForcef", runTime.timeName(), mesh),
    mesh,
    dimensionedScalar(dimForce/dimVolume, Zero)
);

// ---- static per-face factors (mesh does not move) -------------------------
// Kf  = -F0*S(d)*tanh(d/d0)*(dhat & nf)        mean EHD force, times (1-alpha_f)
// Sfl = 0.5*(1 - tanh((d - Rfl)/wE))           window of the fluctuating stress
// cosT[k], sinT[k] = cos/sin(m theta)          angular modes
auto plFaceStatic = [&](const vector& c, const vector& sf, scalar msf,
                        scalar& K, scalar& Sw, scalar& th)
{
    const vector dv = c - plX0;
    const scalar d = mag(dv) + VSMALL;
    const vector nf = sf/(msf + VSMALL);
    const scalar S = 0.5*(1.0 - Foam::tanh((d - plRE)/plwE));
    K = -plF0*S*Foam::tanh(d/pld0)*((dv/d) & nf);
    Sw = 0.5*(1.0 - Foam::tanh((d - plRfl)/plwE));
    const scalar za = (dv & plAx);
    th = Foam::atan2(mag(dv - za*plAx), za);
};

const label plNf = mesh.nFaces();
scalarField plK(plNf), plSw(plNf), plTh(plNf);
{
    const surfaceVectorField& Cf = mesh.Cf();
    const surfaceVectorField& Sf = mesh.Sf();
    const surfaceScalarField& magSf = mesh.magSf();
    label n = 0;
    forAll(Cf, i)
    {
        plFaceStatic(Cf[i], Sf[i], magSf[i], plK[n], plSw[n], plTh[n]); ++n;
    }
    forAll(Cf.boundaryField(), p)
    {
        const fvsPatchVectorField& pc = Cf.boundaryField()[p];
        forAll(pc, i)
        {
            plFaceStatic(pc[i], Sf.boundaryField()[p][i], magSf.boundaryField()[p][i],
                         plK[n], plSw[n], plTh[n]);
            ++n;
        }
    }
    plK.setSize(n); plSw.setSize(n); plTh.setSize(n);
}
scalarField plUp(plPup*plSw*pow(max(cos(plTh), scalar(0)), plnUp));
PtrList<scalarField> plCos(plNm), plSin(plNm);
for (label k = 0; k < plNm; ++k)
{
    const scalar m = plM0 + k;
    plCos.set(k, new scalarField(cos(m*plTh)));
    plSin.set(k, new scalarField(sin(m*plTh)));
}

// cell centres for post-processing
{
    volVectorField cc("C", mesh.C());
    cc.write();
}
