{
    // advance the OU amplitudes
    const scalar dtNow = runTime.deltaTValue();
    const scalar ex = Foam::exp(-dtNow/plTau);
    const scalar sq = Foam::sqrt(max(1.0 - ex*ex, 0.0));
    forAll(plA, k)
    {
        plA[k] = ex*plA[k] + sq*plRnd.GaussNormal<scalar>();
        plB[k] = ex*plB[k] + sq*plRnd.GaussNormal<scalar>();
    }
    const scalar norm = 1.0/Foam::sqrt(scalar(plNm));

    const surfaceScalarField alpha1f(fvc::interpolate(alpha1));
    const surfaceScalarField snGradAlpha(fvc::snGrad(alpha1));

    // fluctuating stress amplitude per face
    scalarField xi(plK.size(), Zero);
    if (plPfl > 0)
    {
        for (label k = 0; k < plNm; ++k)
        {
            xi += plA[k]*plCos[k] + plB[k]*plSin[k];
        }
        xi *= plPfl*norm*plSw;
    }

    label n = 0;
    scalarField& pf = plasmaForcef.primitiveFieldRef();
    forAll(pf, i)
    {
        pf[i] = plK[n]*(1.0 - alpha1f[i]) - xi[n]*snGradAlpha[i];
        ++n;
    }
    auto& bf = plasmaForcef.boundaryFieldRef();
    forAll(bf, p)
    {
        fvsPatchScalarField& pp = bf[p];
        const fvsPatchScalarField& pa = alpha1f.boundaryField()[p];
        const fvsPatchScalarField& pg = snGradAlpha.boundaryField()[p];
        forAll(pp, i)
        {
            pp[i] = plK[n]*(1.0 - pa[i]) - xi[n]*pg[i];
            ++n;
        }
    }
}
