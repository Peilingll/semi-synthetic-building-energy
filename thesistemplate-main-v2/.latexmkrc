use File::Basename;

# Portable MiKTeX on D: -- adjust if the installation moves.
my $MIKTEX = 'D:/MiKTeX/texmfs/install/miktex/bin/x64';

$pdf_mode  = 1;
$pdflatex  = "$MIKTEX/pdflatex -interaction=nonstopmode -synctex=1 --enable-installer %O %S";
$biber     = "$MIKTEX/biber %O %S";
$bibtex_use = 2;    # run biber and clean its output

# glossaries-extra writes .glo-abr for the abbreviations list; teach latexmk
# to regenerate it so the List of Abbreviations stays in sync on a rebuild.
sub run_makeglossaries {
    my ($base, $path) = fileparse($_[0]);
    pushd $path;
    my $ret = system("$MIKTEX/makeglossaries", $base);
    popd;
    return $ret;
}
add_cus_dep('glo-abr', 'gls-abr', 0, 'run_makeglossaries');
add_cus_dep('glo', 'gls', 0, 'run_makeglossaries');
add_cus_dep('slo', 'sls', 0, 'run_makeglossaries');

push @generated_exts, 'glo', 'gls', 'glg', 'glo-abr', 'gls-abr', 'glg-abr',
                      'slo', 'sls', 'slg', 'glsdefs', 'ist', 'run.xml', 'bcf';
