# Citing active_subspaces

If `active_subspaces` contributes to work you publish, please cite the software paper, and the
methods papers for the methods you use.

## The software

Until a paper on version 1.0 exists, the preferred citation is the 2016 software paper
(Constantine et al., 2016):

> P. Constantine, R. Howard, A. Glaws, Z. Grey, P. Diaz, L. Fletcher. Python Active-subspaces
> Utility Library. *Journal of Open Source Software* 1(5):79, 2016.
> [doi:10.21105/joss.00079](https://doi.org/10.21105/joss.00079)

```bibtex
@article{Constantine2016,
  author  = {Constantine, Paul and Howard, Ryan and Glaws, Andrew and Grey, Zachary and
             Diaz, Paul and Fletcher, Leslie},
  title   = {Python Active-subspaces Utility Library},
  journal = {Journal of Open Source Software},
  volume  = {1},
  number  = {5},
  pages   = {79},
  year    = {2016},
  doi     = {10.21105/joss.00079}
}
```

To cite the exact version of the code you ran, cite its software archive on Zenodo as well.
The 2016 release is archived as [doi:10.5281/zenodo.158941](https://doi.org/10.5281/zenodo.158941).
Version 1.0 and later releases will be archived as new Zenodo versions; their DOIs are not
assigned yet (to verify at release).

The repository's `CITATION.cff` holds the same information in machine-readable form, with the
full list of software authors. GitHub's "Cite this repository" button and reference managers read
it:

```{literalinclude} ../CITATION.cff
:language: yaml
```

## The methods

Active subspaces are defined and analyzed in Constantine (2015) and in Constantine, Dow and Wang
(2014). The API reference and the theory pages cite the papers behind each estimator, dimension
rule and algorithm; please cite those you use.

## References

- Constantine, P. G. (2015). *Active Subspaces: Emerging Ideas for Dimension Reduction in
  Parameter Studies.* SIAM. [doi:10.1137/1.9781611973860](https://doi.org/10.1137/1.9781611973860)
- Constantine, P. G., Dow, E., Wang, Q. (2014). Active subspace methods in theory and practice:
  applications to kriging surfaces. *SIAM Journal on Scientific Computing* 36(4):A1500–A1524.
  [doi:10.1137/130916138](https://doi.org/10.1137/130916138)
- Constantine, P., Howard, R., Glaws, A., Grey, Z., Diaz, P., Fletcher, L. (2016). Python
  Active-subspaces Utility Library. *Journal of Open Source Software* 1(5):79.
  [doi:10.21105/joss.00079](https://doi.org/10.21105/joss.00079)
