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

    // ---- gas connected to the discharge point (the plasma lives only there) ----
    // flood fill over gas cells (alpha1 < 0.5) from the cells near x0, then grow
    // the region by plNlayer cell layers so that the interface cells are included
    const labelListList& cc = mesh.cellCells();
    const scalarField& a1 = alpha1.primitiveField();
    const vectorField& C = mesh.C().primitiveField();
    boolList conn(mesh.nCells(), false);
    {
        DynamicList<label> front;
        forAll(C, ci)
        {
            if (a1[ci] < 0.5 && mag(C[ci] - plX0) < plSeedR)
            {
                conn[ci] = true;
                front.append(ci);
            }
        }
        while (front.size())
        {
            DynamicList<label> next;
            forAll(front, k)
            {
                const labelList& nb = cc[front[k]];
                forAll(nb, j)
                {
                    const label c = nb[j];
                    if (!conn[c] && a1[c] < 0.5)
                    {
                        conn[c] = true;
                        next.append(c);
                    }
                }
            }
            front.transfer(next);
        }
        for (label layer = 0; layer < plNlayer; ++layer)
        {
            boolList grown(conn);
            forAll(conn, ci)
            {
                if (conn[ci])
                {
                    const labelList& nb = cc[ci];
                    forAll(nb, j) { grown[nb[j]] = true; }
                }
            }
            conn.transfer(grown);
        }
    }
    const labelUList& own = mesh.owner();
    const labelUList& nei = mesh.neighbour();

    const surfaceScalarField alpha1f(fvc::interpolate(alpha1));
    const surfaceScalarField snGradAlpha(fvc::snGrad(alpha1));

    // fluctuating stress amplitude per face
    scalarField xi(plUp);
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
        const scalar mk = (conn[own[i]] || conn[nei[i]]) ? 1.0 : 0.0;
        pf[i] = mk*(plK[n]*(1.0 - alpha1f[i]) + xi[n]*snGradAlpha[i]);
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
            const scalar mk = conn[pp.patch().faceCells()[i]] ? 1.0 : 0.0;
            pp[i] = mk*(plK[n]*(1.0 - pa[i]) + xi[n]*pg[i]);
            ++n;
        }
    }
}
