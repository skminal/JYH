#!/usr/bin/env python3
"""Generate an OpenFOAM (v1912) case: helium plasma bubble in ethanol.

Set-up follows Wang et al., Matter 7 (2024) 3024-3035:
  silica capillary ID 0.3 mm / OD 0.5 mm, He 3 mL/min, anhydrous ethanol
  (mu 1.17 mPa s, rho 791 kg/m3, sigma 22.14 mN/m), ring electrode 30 mm
  above, DC 2.0 kV.  Axisymmetric wedge, axis = z, orifice at z = 0.

The plasma is represented by the phenomenological terms of plasmaInterFoam
(constant/plasmaProperties): a central EHD force on the gas toward the
discharge point and a fluctuating interfacial stress whose amplitude encodes
the polarity.  Surface tension (CSF) is always on.

usage: make_case.py OUTDIR --mode {off,pos,neg} [options]
"""
import argparse
import json
import math
import os

R_IN, R_OUT = 0.15e-3, 0.25e-3      # capillary radii
Q = 3e-6 / 60.0                     # 3 mL/min
RHO_L, MU_L, SIGMA = 791.0, 1.17e-3, 22.14e-3
RHO_G, MU_G = 0.1625, 1.99e-5       # helium, 25 C, 1 atm
G = 9.81
WEDGE = 2.5                         # half angle, degrees

HDR = '''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: plasma bubble case (generated)
   \\\\    /   O peration     |
    \\\\  /    A nd           |
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       %s;
    location    "%s";
    object      %s;
}
'''


def w(path, cls, loc, obj, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(HDR % (cls, loc, obj) + body)


def block_mesh(dx):
    """Blocks in (r, z). The capillary wall r in [R_IN, R_OUT], z < 0 is left out."""
    mm = 1e-3
    rs = [0.0, R_IN, R_OUT, 2.5 * mm, 5.0 * mm]
    zs = [-3.0 * mm, -2.0 * mm, 0.0, 7.0 * mm, 20.0 * mm]
    nr = [round(R_IN / dx), round((R_OUT - R_IN) / dx), round((2.5 * mm - R_OUT) / dx), 28]
    nz = [12, round(2.0 * mm / dx), round(7.0 * mm / dx), 75]
    gr = [1, 1, 1, 6.0]          # radial grading of the outer column
    gz = [0.2, 1, 1, 7.0]        # axial grading: fine toward z = -2 mm and z = 7 mm
    c, s = math.cos(math.radians(WEDGE)), math.sin(math.radians(WEDGE))
    verts, vid = [], {}

    def v(i, j, side):
        key = (i, j, side if i > 0 else 0)
        if key in vid:
            return vid[key]
        r, z = rs[i], zs[j]
        y = -r * s if side == 0 else r * s
        verts.append((r * c, y, z))
        vid[key] = len(verts) - 1
        return vid[key]

    blocks = []
    skip = {(1, 0), (1, 1)}              # capillary wall
    for j in range(4):
        for i in range(4):
            if (i, j) in skip:
                continue
            # local x1 = r, x2 = wedge (side 0 -> 1, +y), x3 = z : right-handed
            a = [v(i, j, 0), v(i + 1, j, 0), v(i + 1, j, 1), v(i, j, 1),
                 v(i, j + 1, 0), v(i + 1, j + 1, 0), v(i + 1, j + 1, 1), v(i, j + 1, 1)]
            blocks.append('    hex (%s) (%d 1 %d) simpleGrading (%g 1 %g)'
                          % (' '.join(map(str, a)), nr[i], nz[j], gr[i], gz[j]))

    def face(i0, j0, i1, j1, front=True):
        # quad in the r-z plane edge -> the face spanning the wedge
        return '(%d %d %d %d)' % (v(i0, j0, 0), v(i1, j1, 0), v(i1, j1, 1), v(i0, j0, 1))

    patches = {
        'inlet': ('patch', [face(0, 0, 1, 0)]),
        'capillary': ('wall', [face(1, 0, 1, 1), face(1, 1, 1, 2),        # inner wall
                               face(2, 0, 2, 1), face(2, 1, 2, 2),        # outer wall
                               face(1, 2, 2, 2)]),                        # rim at z = 0
        'bottom': ('wall', [face(2, 0, 3, 0), face(3, 0, 4, 0)]),
        'outer': ('patch', [face(4, j, 4, j + 1) for j in range(4)]),
        'top': ('patch', [face(i, 4, i + 1, 4) for i in range(4)]),
    }
    # wedge faces
    front, back = [], []
    for j in range(4):
        for i in range(4):
            if (i, j) in skip:
                continue
            front.append('(%d %d %d %d)' % (v(i, j, 0), v(i, j + 1, 0), v(i + 1, j + 1, 0), v(i + 1, j, 0)))
            back.append('(%d %d %d %d)' % (v(i, j, 1), v(i + 1, j, 1), v(i + 1, j + 1, 1), v(i, j + 1, 1)))
    patches['front'] = ('wedge', front)
    patches['back'] = ('wedge', back)
    patches['axis'] = ('empty', [])

    body = 'scale 1;\n\nvertices\n(\n%s\n);\n\nblocks\n(\n%s\n);\n\nedges ();\n\nboundary\n(\n' % (
        '\n'.join('    (%.10g %.10g %.10g)' % p for p in verts), '\n'.join(blocks))
    for name, (typ, faces) in patches.items():
        body += '    %s\n    {\n        type %s;\n        faces\n        (\n%s\n        );\n    }\n' % (
            name, typ, '\n'.join('            ' + f for f in faces))
    body += ');\n\nmergePatchPairs ();\n'
    ncell = sum(nr[i] * nz[j] for j in range(4) for i in range(4) if (i, j) not in skip)
    return body, ncell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--mode', choices=['off', 'pos', 'neg'], required=True)
    ap.add_argument('--dx', type=float, default=25e-6)
    ap.add_argument('--kappa', type=float, default=14.7, help='F0 / (rho_l g)')
    ap.add_argument('--RE', type=float, default=1.7e-3)
    ap.add_argument('--wE', type=float, default=0.2e-3)
    ap.add_argument('--Ppos', type=float, default=40.0)
    ap.add_argument('--Pneg', type=float, default=500.0)
    ap.add_argument('--Rb0', type=float, default=1.47e-3, help='initial (remnant) bubble radius')
    ap.add_argument('--zb0', type=float, default=0.10e-3, help='initial bubble centre height')
    ap.add_argument('--tend', type=float, default=0.10)
    ap.add_argument('--dtw', type=float, default=0.5e-3)
    ap.add_argument('--theta', type=float, default=45.0, help='contact angle on glass, through ethanol')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--Pup', type=float, default=0.0, help='upward (ion-wind) interfacial stress, Pa')
    ap.add_argument('--nUp', type=float, default=2.0)
    a = ap.parse_args()
    o = a.out

    U_in = Q / (math.pi * R_IN**2)
    F0 = 0.0 if a.mode == 'off' else a.kappa * RHO_L * G
    Pfl = {'off': 0.0, 'pos': a.Ppos, 'neg': a.Pneg}[a.mode]

    bm, ncell = block_mesh(a.dx)
    w(o + '/system/blockMeshDict', 'dictionary', 'system', 'blockMeshDict', bm)

    w(o + '/system/controlDict', 'dictionary', 'system', 'controlDict', '''
application     plasmaInterFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         %g;
deltaT          1e-7;
writeControl    adjustableRunTime;
writeInterval   %g;
purgeWrite      0;
writeFormat     ascii;
writePrecision  7;
writeCompression off;
timeFormat      general;
timePrecision   8;
runTimeModifiable yes;
adjustTimeStep  yes;
maxCo           0.4;
maxAlphaCo      0.4;
maxDeltaT       2e-5;
''' % (a.tend, a.dtw))

    w(o + '/system/fvSchemes', 'dictionary', 'system', 'fvSchemes', '''
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    div(rhoPhi,U)  Gauss linearUpwind grad(U);
    div(phi,alpha) Gauss vanLeer;
    div(phirb,alpha) Gauss linear;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
''')

    w(o + '/system/fvSolution', 'dictionary', 'system', 'fvSolution', '''
solvers
{
    "alpha.ethanol.*"
    {
        nAlphaCorr      2;
        nAlphaSubCycles 1;
        cAlpha          1;
        MULESCorr       yes;
        nLimiterIter    3;
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0;
    }
    "pcorr.*"   { solver PCG; preconditioner DIC; tolerance 1e-5; relTol 0; }
    p_rgh       { solver GAMG; smoother DIC; tolerance 1e-7; relTol 0.01; }
    p_rghFinal  { $p_rgh; relTol 0; }
    U           { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-6; relTol 0; }
    UFinal      { $U; }
}
PIMPLE
{
    momentumPredictor   no;
    nOuterCorrectors    1;
    nCorrectors         3;
    nNonOrthogonalCorrectors 0;
}
relaxationFactors { equations { ".*" 1; } }
''')

    gas_r = a.Rb0
    w(o + '/system/setFieldsDict', 'dictionary', 'system', 'setFieldsDict', '''
defaultFieldValues ( volScalarFieldValue alpha.ethanol 1 );
regions
(
    sphereToCell
    {
        centre (0 0 %g);
        radius %g;
        fieldValues ( volScalarFieldValue alpha.ethanol 0 );
    }
    cylinderToCell
    {
        p1 (0 0 -0.004);
        p2 (0 0 0);
        radius %g;
        fieldValues ( volScalarFieldValue alpha.ethanol 0 );
    }
);
''' % (a.zb0, gas_r, R_IN * 1.01))

    w(o + '/constant/g', 'uniformDimensionedVectorField', 'constant', 'g',
      '\ndimensions [0 1 -2 0 0 0 0];\nvalue (0 0 -%g);\n' % G)
    w(o + '/constant/transportProperties', 'dictionary', 'constant', 'transportProperties', '''
phases (ethanol helium);
ethanol { transportModel Newtonian; nu %g; rho %g; }
helium  { transportModel Newtonian; nu %g; rho %g; }
sigma %g;
''' % (MU_L / RHO_L, RHO_L, MU_G / RHO_G, RHO_G, SIGMA))
    w(o + '/constant/turbulenceProperties', 'dictionary', 'constant', 'turbulenceProperties',
      '\nsimulationType laminar;\n')
    w(o + '/constant/dynamicMeshDict', 'dictionary', 'constant', 'dynamicMeshDict',
      '\ndynamicFvMesh staticFvMesh;\n')
    w(o + '/constant/plasmaProperties', 'dictionary', 'constant', 'plasmaProperties', '''
F0      %g;        // N/m3  (= %g rho_l g)
RE      %g;
wE      %g;
d0      5e-05;
x0      (0 0 0);
axis    (0 0 1);
Pfl     %g;        // Pa
Rfl     4e-03;
tau     3e-04;
mMin    4;
mMax    24;
seed    %d;
Pup     %g;        // Pa
nUp     %g;
''' % (F0, a.kappa if F0 else 0, a.RE, a.wE, Pfl, a.seed, a.Pup if F0 else 0.0, a.nUp))

    bnd_common = '''
    front { type wedge; }
    back  { type wedge; }
    axis  { type empty; }
'''
    w(o + '/0/U', 'volVectorField', '0', 'U', '''
dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{
    inlet     { type fixedValue; value uniform (0 0 %g); }
    capillary { type noSlip; }
    bottom    { type noSlip; }
    outer     { type slip; }
    top       { type pressureInletOutletVelocity; value uniform (0 0 0); }
%s}
''' % (U_in, bnd_common))
    w(o + '/0/p_rgh', 'volScalarField', '0', 'p_rgh', '''
dimensions [1 -1 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet     { type fixedFluxPressure; value uniform 0; }
    capillary { type fixedFluxPressure; value uniform 0; }
    bottom    { type fixedFluxPressure; value uniform 0; }
    outer     { type fixedFluxPressure; value uniform 0; }
    top       { type totalPressure; p0 uniform 0; value uniform 0; }
%s}
''' % bnd_common)
    w(o + '/0/alpha.ethanol.orig', 'volScalarField', '0', 'alpha.ethanol', '''
dimensions [0 0 0 0 0 0 0];
internalField uniform 1;
boundaryField
{
    inlet     { type fixedValue; value uniform 0; }
    capillary { type constantAlphaContactAngle; theta0 %g; limit gradient; value uniform 1; }
    bottom    { type zeroGradient; }
    outer     { type zeroGradient; }
    top       { type inletOutlet; inletValue uniform 1; value uniform 1; }
%s}
''' % (a.theta, bnd_common))

    meta = dict(mode=a.mode, dx=a.dx, ncell=ncell, U_in=U_in, Q=Q, F0=F0, kappa=a.kappa if F0 else 0,
                RE=a.RE, wE=a.wE, Pfl=Pfl, Pup=a.Pup if F0 else 0.0, nUp=a.nUp, Rb0=a.Rb0, zb0=a.zb0, tend=a.tend, dtw=a.dtw, theta=a.theta,
                rho_l=RHO_L, mu_l=MU_L, sigma=SIGMA, rho_g=RHO_G, mu_g=MU_G, R_in=R_IN, R_out=R_OUT)
    json.dump(meta, open(o + '/case.json', 'w'), indent=1)
    print('wrote', o, 'cells ~', ncell, 'U_in %.3f m/s' % U_in, 'F0 %.4g N/m3' % F0, 'Pfl', Pfl)


if __name__ == '__main__':
    main()
