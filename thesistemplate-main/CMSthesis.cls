\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{CMSthesis}[2022/12/12 CMS, TU Munich thesis class]

\RequirePackage{iftex}
\RequirePackage{etoolbox}
\RequirePackage{kvoptions}
\RequirePackage{xstring}

\SetupKeyvalOptions{family=BGU,
                    prefix=BGU@,
                    }
\DeclareBoolOption[false]{showframe}
\DeclareBoolOption[false]{print}
\DeclareBoolOption[true]{colorcode}
\DeclareBoolOption[true]{microtype}
\DeclareStringOption[ngerman]{language}
\DeclareStringOption[apa]{citeStyle}
\DeclareBoolOption[true]{thesis}
\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessKeyvalOptions*

\LoadClass[a4paper]{book}

\newcommand{\hrefopt}{}
\ifboolexpr{not (bool{XeTeX} or bool{LuaTeX})}{%
    \RequirePackage[T1]{fontenc}
	\RequirePackage[utf8]{inputenc}
    \RequirePackage{helvet}
    \renewcommand{\familydefault}{\sfdefault}
    \ifbool{PDFTeX}{
        \pdfcompresslevel=1
        \pdfimageresolution=300
        \renewcommand{\hrefopt}{pdftex}
    }{
        \renewcommand{\hrefopt}{dvips}
        \PassOptionsToPackage{dvips}{graphicx}
    }
}{
    \RequirePackage{fontspec}
	\setmainfont{Arial}
	\ifbool{XeTeX}{\renewcommand{\hrefopt}{xetex}}{}
}

\newcommand{\DefineColor}[4][0]{%
    \ifboolexpr{bool{BGU@colorcode} and not bool{BGU@print}}{%
    \IfStrEq{#3}{mix}{
        \colorlet{#2}{#4}
    }{
        \definecolor{#2}{#3}{#4}
    }
    }{%
    \definecolor{#2}{gray}{#1}
    }
}

\newbool{ger}
\IfStrEq{\BGU@language}{ngerman}{\booltrue{ger}}{}

\RequirePackage[dvipsnames]{xcolor}
\definecolor{linkblue}{rgb}{0,0.1,0.6}
\definecolor{citegreen}{rgb}{0,0.25,0.15}
\definecolor{linkred}{rgb}{0.8,0,0.005}
\definecolor{mailviolet}{rgb}{0.3,0,0.35}
\definecolor{tumblue}{rgb}{0,0.396,0.741}

\DefineColor[0.1]{kwcolor}{mix}{blue!70!black}
\DefineColor[0.8]{kwcolor2}{mix}{magenta!80!white}
\DefineColor[0.3]{kwcolor3}{mix}{OliveGreen}
\DefineColor[0.6]{kwcolor4}{mix}{Yellow!50!OliveGreen}
\DefineColor[0.7]{kwcolor5}{mix}{blue!50!white}
\DefineColor[0.4]{kwcolor6}{mix}{RedViolet}
\DefineColor[0.4]{commentcolor}{mix}{green!50!black}
\DefineColor[0.5]{commentcolor2}{mix}{gray}
\DefineColor[0.1]{stringcolor}{mix}{violet!50!black}

\ifger
\RequirePackage[english, ngerman]{babel}
\gdef\DAchair{Lehrstuhl für Computing in Civil and Building Engineering}
\else
\RequirePackage[ngerman, english]{babel}
\gdef\DAchair{Chair of Computing in Civil and Building Engineering}
\fi

\newcommand*{\DAfaculty}{TUM School of Engineering and Design}
\newcommand*{\DAtum}{}
\newcommand*{\TPauthor}{}
\newcommand*{\TPmatrNo}{}
\newcommand*{\TPsupervisor}{}
\newcommand*{\TPstartDate}{}
\newcommand*{\TPendDate}{}
\newcommand*{\DPdeclaration}{}
\newcommand*{\DPtext}{}
\newcommand*{\DPtextOld}{}
\newcommand*{\DPsign}{}
\newcommand*{\TPtext}{}
\newcommand*{\tumAddress}{}
\newcommand*{\Symblistname}{}
\newcommand*{\rtitle}{}

\addto\captionsngerman{\renewcommand{\lstlistingname}{Algorithmus}}
\addto\captionsngerman{\renewcommand{\lstlistlistingname}{Algorithmenverzeichnis}}
\addto\captionsngerman{\crefname{listing}{Algorithmus}{Algorithmen}}
\addto\captionsngerman{\Crefname{listing}{Algorithmus}{Algorithmen}}
\addto\captionsngerman{\renewcommand{\DAtum}{Technische Universität München}}
\addto\captionsngerman{\renewcommand{\TPauthor}{{\bfseries Eingereicht von}}}
\addto\captionsngerman{\renewcommand{\TPmatrNo}{Matrikelnummer}}
\addto\captionsngerman{\renewcommand{\TPsupervisor}{{\bfseries Betreut von}}}
\addto\captionsngerman{\renewcommand{\TPstartDate}{Ausgabedatum}}
\addto\captionsngerman{\renewcommand{\TPendDate}{{\bfseries Eingereicht am}}}
\addto\captionsngerman{\renewcommand{\DPdeclaration}{Eidesstattliche Erklärung}}
\addto\captionsngerman{\renewcommand{\Symblistname}{Symbolverzeichnis}}
\addto\captionsngerman{\renewcommand{\rtitle}{Literaturverzeichnis}}
\addto\captionsngerman{\renewcommand{\DPtext}{%
Ich versichere hiermit, dass ich die von mir eingereichte Abschlussarbeit selbständig verfasst und keine anderen als die angegebenen Quellen und Hilfsmittel benutzt habe.}}
\addto\captionsngerman{\renewcommand{\DPsign}{Ort, Datum, Unterschrift}}
\addto\captionsngerman{\renewcommand{\TPtext}{%
	{\Large
    	Wissenschaftliche Arbeit zur Erlangung des Grades\vskip.5ex
    	{\bf{\DAtype}}\vskip1ex
    	an der \DAfaculty{}\\
        der Technischen Universität München.
    }
    }}
\addto\captionsngerman{\renewcommand{\tumAddress}{Arcisstr. 21\\D-80333 München}}
\addto\captionsngerman{\DefTblrTemplate{contfoot-text}{default}{Fortsetzung auf der nächsten Seite}}
\addto\captionsngerman{\DefTblrTemplate{conthead-text}{default}{(Fortsetzung)}}

\addto\captionsenglish{\renewcommand{\lstlistingname}{Algorithm}}
\addto\captionsenglish{\renewcommand{\lstlistlistingname}{List of Algorithms}}
\addto\captionsenglish{\crefname{listing}{Algorithm}{Algorithms}}
\addto\captionsenglish{\Crefname{listing}{Algorithm}{Algorithms}}
\addto\captionsenglish{\renewcommand{\DAtum}{Technical University of Munich}}
\addto\captionsenglish{\renewcommand{\TPauthor}{{\bfseries Submitted by}}}
\addto\captionsenglish{\renewcommand{\TPmatrNo}{Student ID}}
\addto\captionsenglish{\renewcommand{\TPsupervisor}{{\bfseries Supervised by}}}
\addto\captionsenglish{\renewcommand{\TPstartDate}{Date of Issue}}
\addto\captionsenglish{\renewcommand{\TPendDate}{{\bfseries Submitted on}}}
\addto\captionsenglish{\renewcommand{\DPdeclaration}{Declaration}}
\addto\captionsenglish{\renewcommand{\Symblistname}{List of Symbols}}
\addto\captionsenglish{\renewcommand{\rtitle}{References}}
\addto\captionsenglish{\renewcommand{\DPtext}{%
	I hereby affirm that I have independently written the thesis submitted by me and have not used any sources or aids other than those indicated.}}
\addto\captionsenglish{\renewcommand{\DPsign}{Location, Date, Signature}}
\addto\captionsenglish{\renewcommand{\TPtext}{%
	{\Large
	Scientific work to obtain the degree\vskip.5ex
	{\bf{\DAtype}}\vskip1ex
	at the \DAfaculty{}\\ 
	of the Technical University of Munich.}
    }}
\addto\captionsenglish{\renewcommand{\tumAddress}{Arcisstr. 21\\D-80333 Munich}}

\renewcommand{\ttdefault}{pcr}
\RequirePackage{listings}
\lstset{%
	mathescape=true,
	frame=tB,
	numberstyle=\tiny,
	escapeinside={(*@}{@*)},
	basicstyle=\small\ttfamily, 
	keywordstyle=\color{kwcolor}\bfseries\em,
	keywords={for, if, else, foreach, while, end, do, procedure, to, return, break, continue },
	extendedchars=true,
	literate={ä}{{\"a}}1 {ö}{{\"o}}1 {ü}{{\"u}}1 {ß}{{\ss}}1,
	breaklines=true,
}
%% inserted code with Copyright
%% Copyright (C) 2014 by Julien Cretel <jubobs.tex at gmail.com>
%% 
%% This work may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.3
%% of this license or (at your option) any later version.
\lst@Key{autodedent}f[t]{\lstKV@SetIf{#1}\ifautodedent@lstad@}

\newtoks\toks@lstad
\toks@lstad={%
    \lst@DefSaveDef{`\ }\space@lstad{\processSpace@lstad}%
    %\lst@DefSaveDef{`\^^I}\tab@lstad{\processTab@lstad} % currently not used
}

\newcommand\add@savedef@lstad{}
\def\add@savedef@lstad#1#2{%
  \begingroup\lccode`?=#1\relax
  \lowercase{\endgroup
    \edef\@tempa{%
      \noexpand\lst@DefSaveDef{\number#1}%
      \expandafter\noexpand\csname ?@lstad\endcsname{\noexpand#2%
      \expandafter\noexpand\csname ?@lstad\endcsname}%
    }%
  }%
  \toks@lstad=\expandafter{\the\expandafter\toks@lstad\@tempa}%
}

\count@=33
\loop
    \add@savedef@lstad\count@\processchar@lstad
  \ifnum\count@<127
  \advance\count@\@ne
\repeat

\newcount\spacesToGobble@lstad
\newcount\spacesGobbledSoFar@lstad
\newif\ifafterindent@lstad@
\newif\ifonBasisLine@lstad@
\newif\ifPostponeCount@lstad@

\lst@AddToHook{Init}
{%
  \ifautodedent@lstad@%
    % patch hook macros
    \let\@ddedToInitVarsBOLhook@lstad\@@ddedToInitVarsBOLhook@lstad%
    \let\@ddedToEOLhook@lstad\@@ddedToEOLhook@lstad%
    % initialise things
    \global\onBasisLine@lstad@true%
    \global\afterindent@lstad@false%
    \global\spacesToGobble@lstad=0%
    \lst@ifincluderangemarker%
      % if firtline option is not used
      \ifnum9999999=\lst@firstline%
        \global\PostponeCount@lstad@true%
      \fi
    \fi
  \fi
}

\lst@AddToHook{SelectCharTable}
{%
  \ifautodedent@lstad@%
    \lst@lAddTo\lst@DeveloperSCT{\the\toks@lstad}%
  \fi
}

\lst@AddToHook{InitVarsBOL}{\@ddedToInitVarsBOLhook@lstad}
\newcommand\@ddedToInitVarsBOLhook@lstad{}
\newcommand\@@ddedToInitVarsBOLhook@lstad{\global\afterindent@lstad@false}

\lst@AddToHook{EOL}{\@ddedToEOLhook@lstad}
\newcommand\@ddedToEOLhook@lstad{}
\newcommand\@@ddedToEOLhook@lstad
{%
  \ifonBasisLine@lstad@%
    \ifPostponeCount@lstad@%
    \else
      \global\onBasisLine@lstad@false%
    \fi
  \fi
  \global\spacesGobbledSoFar@lstad=0%
  \global\PostponeCount@lstad@false%
}

\lst@AddToHook{DeInit}
{%
  \ifautodedent@lstad@%
    % undo patches
    \renewcommand\@ddedToInitVarsBOLhook@lstad{}%
    \renewcommand\@ddedToEOLhook@lstad{}%
  \fi
}

\newcommand\processchar@lstad{\global\afterindent@lstad@true}

\newcommand\processSpace@lstad
{%
  \ifafterindent@lstad@%
    \space@lstad%
  \else
    \ifonBasisLine@lstad@%
      \global\advance\spacesToGobble@lstad\@ne\relax%
    \else
      \ifnum\spacesGobbledSoFar@lstad<\spacesToGobble@lstad\relax%
        \global\advance\spacesGobbledSoFar@lstad\@ne\relax%
      \else
        \space@lstad%
      \fi
    \fi
  \fi
}
%% end of Copyright

\def\lst@OpLiteratekey#1\@nil@{\let\lst@ifxopliterate\lst@if
	\def\lst@opliterate{#1}}
\lst@Key{opliterate}{}{\@ifstar{\lst@true \lst@OpLiteratekey}
	{\lst@false\lst@OpLiteratekey}#1\@nil@}
\lst@AddToHook{SelectCharTable}
{\ifx\lst@opliterate\@empty\else
	\expandafter\lst@OpLiterate\lst@opliterate{}\relax\z@
	\fi}
\def\lst@OpLiterate#1#2#3{%
	\ifx\relax#2\@empty\else
	\lst@CArgX #1\relax\lst@CDef
	{}
	{\let\lst@next\@empty
		\lst@ifxopliterate
		\lst@ifmode \let\lst@next\lst@CArgEmpty \fi
		\fi
		\ifx\lst@next\@empty
		\ifx\lst@OutputBox\@gobble\else
		\lst@XPrintToken \let\lst@scanmode\lst@scan@m
		\lst@token{#2}\lst@length#3\relax
		\lst@XPrintToken
		\fi
		\let\lst@next\lst@CArgEmptyGobble
		\fi
		\lst@next}%
	\@empty
	\expandafter\lst@OpLiterate
	\fi}

\RequirePackage{graphicx}
\RequirePackage{tikz}
\RequirePackage{pgfplots}
\pgfplotsset{compat=1.15}

\RequirePackage{rotating}

% Rotate landscape pages in PDF viewers as well as on the printed page.
\RequirePackage{pdflscape}
\RequirePackage{tabularray}
\UseTblrLibrary{booktabs}
\RequirePackage{caption}
\RequirePackage{subcaption}
\captionsetup[table]{skip=0em}

%\RequirePackage{floatrow}
%\floatsetup[table]{capposition=top}
%\DeclareFloatSeparators{centimeter}{\hskip1cm}

\RequirePackage{enumitem}
\RequirePackage{amsmath,amssymb,amsthm,nccmath}
\RequirePackage{framed}
\RequirePackage{footnote}
\RequirePackage{url}
\urlstyle{same}

\IfStrEq{\BGU@citeStyle}{none}{}{
\RequirePackage[style=\BGU@citeStyle,sortcites=true,backend=biber]{biblatex}
\IfStrEq{ieee}{\BGU@citeStyle}{}{
    \DeclareLanguageMapping{\BGU@language}{\BGU@language-\BGU@citeStyle}}
\DeclareBibliographyDriver{standard}{%
  \usebibmacro{bibindex}%
  \usebibmacro{begentry}%
  \usebibmacro{author}:\printfield{year}\printfield[parens]{postfix}%
  \setunit{\labelnamepunct}\newblock
  \usebibmacro{title}%
  \newunit\newblock
  \printfield{number}%
  \setunit{\addspace}\newblock
  \printfield[parens]{type}%
  \newunit\newblock
  \usebibmacro{location+date}%
  \newunit\newblock
  \iftoggle{bbx:url}
    {\usebibmacro{url+urldate}}
    {}%
  \newunit\newblock
  \usebibmacro{addendum+pubstate}%
  \setunit{\bibpagerefpunct}\newblock
  \usebibmacro{pageref}%
  \newunit\newblock
  \usebibmacro{related}%
  \usebibmacro{finentry}}
\DeclareCiteCommand{\citenorm}
  {}
  {\usebibmacro{author}:\bibhyperref{\printfield{year}}\printfield[parens]{postfix}}
  {\multicitedelim}
  {}}

\RequirePackage{titlesec}
\titleformat{\chapter}[display]{\filouter\LARGE\bf}{\chaptertitlename~\thechapter}{.2ex}{\bfseries\huge}
\titlespacing{\chapter}{0pt}{0pt}{2ex}
\RequirePackage[left=4.0cm, right=2cm, top=2cm,bottom=2.5cm]{geometry}

\RequirePackage{setspace}
\onehalfspacing
\parindent 0pt

\RequirePackage[hyperindex,\hrefopt]{hyperref}
\RequirePackage[symbols,abbreviations,nonumberlist,shortcuts]{glossaries-extra}
\renewcommand{\glsnamefont}[1]{\textbf{#1}}
\glsaddkey*{dativ}%
{\glsentrytext{\glslabel}}
{\glsentryld}
{\Glsentryld}
{\glsd}
{\Glsd}
{\GLSd}

\newcommand*{\acd}{\glsd}
\newcommand*{\Acd}{\Glsd}
\newcommand*{\ACd}{\GLSd}

\glsaddkey*{genitiv}%
{\glsentrytext{\glslabel}}
{\glsentrylg}
{\Glsentrylg}
{\glsg}
{\Glsg}
{\GLSg}

\newcommand*{\acg}{\glsg}
\newcommand*{\Acg}{\Glsg}
\newcommand*{\ACg}{\GLSg}

\glsaddstoragekey{cite}{}{\glscite}
\newcommand{\accite}[1]{%
	\def\labeltext{\glscite{#1}}%
	\expandafter\cite{\labeltext}%
}
\newcommand{\acparencite}[1]{%
	\def\labeltext{\glscite{#1}}%
	\expandafter\parencite{\labeltext}%
}
\newcommand{\actextcite}[1]{%
	\def\labeltext{\glscite{#1}}%
	\expandafter\textcite{\labeltext}%
}

\glsaddkey{unit}{\glsentrytext{\glslabel}}{\glsentryunit}{\GLsentryunit}{\glsunit}{\Glsunit}{\GLSunit}

\ifbool{BGU@showframe}{\RequirePackage{showframe}}{}
\ifbool{BGU@print}{%
	\hypersetup{%
		colorlinks=true,
		linkcolor=black,
		citecolor=black,
		urlcolor=black,
		filecolor=black
	}
}{%
	\hypersetup{%
		colorlinks=true,
		linkcolor=tumblue,
		citecolor=citegreen,
		urlcolor=mailviolet,
		filecolor=linkred
	}
}

\ifBGU@microtype
\RequirePackage[final]{microtype}
%% microtype fix for microtype versions older than 2.7
\@ifpackagelater{microtype}{2017/07/07}{}
{
    \def\MT@is@composite#1#2\relax{%
      \ifx\\#2\\\else
        \expandafter\def\expandafter\MT@char\expandafter{\csname\expandafter
                        \string\csname\MT@encoding\endcsname
                        \MT@detokenize@n{#1}-\MT@detokenize@n{#2}\endcsname}%
        \ifx\UnicodeEncodingName\@undefined\else
          \expandafter\expandafter\expandafter\MT@is@uni@comp\MT@char\iffontchar\else\fi\relax
        \fi
        \expandafter\expandafter\expandafter\MT@is@letter\MT@char\relax\relax
        \ifnum\MT@char@ < \z@
          \ifMT@xunicode
            \edef\MT@char{\MT@exp@two@c\MT@strip@prefix\meaning\MT@char>\relax}%
              \expandafter\MT@exp@two@c\expandafter\MT@is@charx\expandafter
                \MT@char\MT@charxstring\relax\relax\relax\relax\relax
          \fi
        \fi
      \fi
    }
    \def\MT@is@uni@comp#1\iffontchar#2\else#3\fi\relax{%
      \ifx\\#2\\\else\edef\MT@char{\iffontchar#2\fi}\fi
    }
}
\else
\fi

\RequirePackage[autostyle,german=guillemets]{csquotes}
\RequirePackage[nameinlink,\BGU@language]{cleveref}
\RequirePackage{hyphenat}

\widowpenalty=10000
\clubpenalty=10000
\tolerance=1000

\gdef\DAtype{Master of Science (M.Sc.)}
\newcommand*{\SetType}[1]{\gdef\DAtype{#1}}

\gdef\DAstudy{XYZ-Ingenieurwesen}
\newcommand*{\SetFieldOfStudy}[1]{\gdef\DAstudy{#1}}

\gdef\DAtopic{Latex-Vorlage für Abschlussarbeiten}
\newcommand*{\SetTopic}[1]{\gdef\DAtopic{#1}}

\gdef\DAsubtopic{}
\newcommand*{\SetSubtopic}[1]{\gdef\DAsubtopic{#1}}

\newcommand*{\SetChair}[1]{\gdef\DAchair{#1}}
\newcommand{\DAchairAdress}{\DAchair\\\DAfaculty\\\DAtum\\\tumAddress}

\gdef\DAauthor{Vorname Name}
\newcommand*{\SetAuthor}[1]{\gdef\DAauthor{#1}}

\gdef\DAmatrNo{1122233}
\newcommand*{\SetMatrikulationNo}[1]{\gdef\DAmatrNo{#1}}

\gdef\DAemail{xyz@tum.de}
\newcommand*{\SetEmail}[1]{\gdef\DAemail{#1}}

\gdef\DAzip{D-12345}
\newcommand*{\SetZipCode}[1]{\gdef\DAzip{#1}}

\gdef\DAlocation{München}
\newcommand*{\SetLocation}[1]{\gdef\DAlocation{#1}}

\gdef\DAstreet{Musterstraße~1}
\newcommand*{\SetStreet}[1]{\gdef\DAstreet{#1}}

\gdef\DAauthorAddress{\parbox{10cm}{\DAstreet\\\DAzip~\DAlocation\\e-Mail: \href{mailto:\DAemail}{\DAemail}}}
\newcommand*{\ResetAddress}[1]{\gdef\DAauthorAddress{#1}}

\gdef\DAstartDate{\today}
\newcommand*{\SetStart}[1]{\gdef\DAstartDate{#1}}

\gdef\DAendDate{\today}
\newcommand*{\SetEnd}[1]{\gdef\DAendDate{#1}}

\gdef\DAacceptDate{TT. Monat JJJJ}
\newcommand*{\SetAcceptDate}[1]{\gdef\DAacceptDate{#1}}

\gdef\DAreferent{}
\newcommand*{\SetReferent}[1]{\gdef\DAreferent{#1}}

\gdef\DAreferentChair{}
\newcommand*{\SetReferentAffil}[1]{\gdef\DAreferentChair{#1}}

\newcommand{\AdaptedReferentChair}{%
    \StrLen{\DAreferentChair}[\ChairLength]%
    \ifnum \ChairLength>60\parbox{.7\linewidth}{\vskip.5ex\DAreferentChair} \else\DAreferentChair \fi%
}

\newcounter{supervisors}
\newcounter{int}
\ifBGU@thesis
	\newcommand\supervisorList{\TPsupervisor}
	\newcommand{\AddSupervisor}[1]{
		\@for\x:={#1}\do{%
			\expandafter\xdef\csname n@super\thesupervisors\endcsname{\x}
			\xappto\supervisorList{\noexpand\> \x\noexpand\\}
			\global\stepcounter{supervisors}
		}
	}
	\newcommand{\AddAffil}[1]{
		\StrLen{#1}[\affilLength]%
		\ifnum \affilLength>60
		\xappto\supervisorList{\noexpand\> \noexpand\parbox{.7\linewidth}{\vskip1ex #1}\noexpand\\[2ex]}
		\else
		\xappto\supervisorList{\noexpand\> #1\noexpand\\[1ex]}
		\fi
	}
\else
	\newcommand\supervisorList{}
	\newcommand{\AddSupervisor}[1]{
		\@for\x:={#1}\do{%
			\expandafter\xdef\csname n@super\thesupervisors\endcsname{\x}
			\ifnum \thesupervisors>1
			\xappto\supervisorList{, }
			\fi
			\xappto\supervisorList{\x\noexpand}
			\global\stepcounter{supervisors}
		}
	}
	\newcommand{\AddAffil}[1]{}
\fi

\RequirePackage{fancyhdr}
\@ifclasswith{book}{twoside}{%
	\newcommand{\fancysetting}{\fancyfoot[EL]{\thepage}
		\fancyfoot[OR]{\thepage}
		\fancyfoot[ER]{}
		\fancyfoot[OL]{}
		\renewcommand{\headrulewidth}{0pt}}
}{
	\newcommand{\fancysetting}{\fancyfoot[L]{}
		\fancyfoot[R]{\thepage}
		\renewcommand{\headrulewidth}{0pt}}
}

\setlength{\headheight}{14pt}
\fancyfootoffset{0cm}
\fancypagestyle{plain}{%
	\fancyhf{}
	\renewcommand{\headrulewidth}{0pt}
	\fancysetting
}

\renewcommand{\chaptermark}[1]{\markboth{\thechapter. #1}{}}
\renewcommand{\sectionmark}[1]{\markright{\thesection. #1}{}}

\fancypagestyle{withHeader}
{%
	\fancyhf{}
    \fancyhead[R]{\includegraphics[height=1.0cm]{./figures/Logo/TUM_Logo_blau_cmyk_pdf.pdf}}
    \fancyhead[L]{\parbox[b][1cm][c]{13cm}{\color{tumblue}{\small\DAchair\\ \DAfaculty\\ \DAtum}}}
}


\newcommand{\titlecontent}{}

\ifBGU@thesis
	\renewcommand{\titlecontent}{%
		{\Huge\bf{\nohyphens\DAtopic}}
		\IfStrEq{\DAsubtopic}{}{}{\vskip1cm{\huge\bf{\DAsubtopic}}}
		\vfill
		\TPtext
		\vspace*{1.5cm}
		
		\begin{tabbing}
			Lehrstuhlinhaber: xxxx   \= Testzeile \kill
			\IfStrEq{\DAreferent}{}{}{%
				\textbf{Referent} \> \DAreferent \\
				\> \AdaptedReferentChair\\[2ex]}%
			\supervisorList
			\TPauthor      	  \> \DAauthor{} \IfStrEq{\DAmatrNo}{}{}{(\DAmatrNo)} \\[1ex]
			\> \DAauthorAddress \\[2ex]
			\TPendDate	      \> \DAendDate
		\end{tabbing}}
\else
	\renewcommand{\titlecontent}{%
		\centering
		{\Huge\bf{\nohyphens\DAtopic}}
		\IfStrEq{\DAsubtopic}{}{}{\vskip1cm{\huge\bf{\DAsubtopic}}}
		\vspace*{1.5cm}
		
		\TPauthor
		
		\DAauthor
		\vfill
		\IfStrEq{\supervisorList}{}{}{
		\TPsupervisor:\ 
		\supervisorList}

        \IfStrEq{\DAendDate}{}{}{
		\TPendDate:\ 
		\DAendDate}
	}
\fi

\renewcommand{\maketitle}{%
\begin{titlepage}
    \renewcommand{\headrulewidth}{0.0pt}
    \renewcommand{\footrulewidth}{0.0pt}
    \newgeometry{left=2cm, right=2cm, top=2cm, bottom=3cm, headsep=18mm, headheight=1.2cm, includeheadfoot}
    \fancyhfoffset{0cm}
    
	\titlecontent
    
    \thispagestyle{withHeader}
\end{titlepage}
\restoregeometry
}

\newcommand{\Declaration}{%
    {
    \chapter*{\DPdeclaration}
    \thispagestyle{empty}
    }
    \DPtext
    \vspace*{2cm}
    
    \rule{\textwidth}{.5pt}\\
    \DPsign
    
    \cleardoublepage
}

\newcommand{\nl}{\hfill\break}
\newcommand{\dC}{$^{\circ}C$}
\newcommand\footnoteref[1]{\protected@xdef\@thefnmark{\ref{#1}}\@footnotemark}

\newcommand{\lquote}[2]{	
	\par
	\begingroup
	\leftskip2em
	\rightskip\leftskip
	\textit{\glqq#1\grqq} \cite{#2}
	\par
	\endgroup
}

\newcommand{\squote}[2]{	
	\textit{\glqq#1\grqq} \cite{#2}
}

\newcommand{\erklaerung}[2]{\par
	\begingroup
	\leftskip=1cm
	\noindent
	\paragraph{#1}
	\label{par:#1}
	#2
	\par
	\endgroup}

\mathchardef\mhyphen="2D
\newcommand{\symbfracunit}[2]{$\left[\frac{#1}{#2}\right]$}
\newcommand{\symbunit}[1]{$\left[#1\right]$}

\renewcommand{\labelitemi}{-}
\renewcommand{\labelitemii}{$\cdot$}
\renewcommand{\labelitemiii}{$\bullet$}
\renewcommand{\labelitemiv}{$\ast$}

\newcommand{\nocontentsline}[3]{}
\newcommand{\tocless}[2]{\bgroup\let\addcontentsline=\nocontentsline#1{#2}\egroup}
