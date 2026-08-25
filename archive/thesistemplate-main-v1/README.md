# thesisTemplate

CMS standard template for typesetting your thesis or report.

## Getting started

For use it is recommended to either download the files as ZIP to upload on Overleaf or create a fork of this project.
This template has been tested with Overleaf and on a local machine using MikTex, but it should also work with TexLive and TUM's own Overleaf server.

In the thesis.tex file specify the type of document by setting the option "thesis" in line 7 to true (for writing a thesis) or false (for writing a report). 
Then fill in your data in the `\Set<...>{}` commands (lines 11&ndash;23).

On a local machine it has to be compiled as follows:
```
pdflatex
biber
makeglossaries-lite 
pdflatex
pdflatex
```
alternatively you can use `makeglossaries` without `-lite`, however, this requires **Perl** to be installed.

## Required packages

In the following you will find a list of all the required latex packages.
The first few are provided with the corresponding link to the CTAN page (you will most likely use these packages when writing your document).

- [listings](https://www.ctan.org/pkg/listings)
- [tikz](https://www.ctan.org/pkg/pgf)
- [rotating](https://www.ctan.org/pkg/rotating)
- [tabularray](https://www.ctan.org/pkg/tabularray)
- [caption](https://www.ctan.org/pkg/caption)
- [glossaries-extra](https://www.ctan.org/pkg/glossaries-extra)
- [cleveref](https://www.ctan.org/pkg/cleveref)
- iftex
- etoolbox
- kvoptions
- xstring
- xcolor
- graphicx
- enumitem
- amsmath,amssymb,amsthm,nccmath
- framed
- url
- pgfplots
- booktabs
- footnote
- subfig
- biblatex
- titlesec
- fancyhdr
- hyperref
- showframe
- microtype
- csquotes
- lscape
