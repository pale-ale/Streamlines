\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mimosis}[2017/08/01 Minimal modern thesis class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrbook}}
\ProcessOptions\relax

\LoadClass[paper=a4,
           twoside,
           pagesize,
           DIV=10,                % TODO: Make configurable
           BCOR=10mm,             % TODO: Make configurable
           cleardoublepage=empty,
           numbers=noenddot,
           titlepage,
           toc=bibliography,
           toc=index,
           ]{scrbook}

\RequirePackage{ifpdf}
\RequirePackage{ifxetex}
\RequirePackage{ifluatex}

\newif\ifxetexorluatex
\ifxetex
  \xetexorluatextrue
\else
  \ifluatex
    \xetexorluatextrue
  \else
    \xetexorluatexfalse
  \fi
\fi

\ifxetexorluatex
  \RequirePackage{fontspec}
\else
  \RequirePackage[utf8]{inputenc}
\fi

% Makes it possible to switch between different languages in the text
% while keeping hyphenation rules correct. Should you add another one
% in the list, please ensure that `english` is the last one. The last
% language is used to control standard hyphenation.
\usepackage[ngerman,french,english]{babel}

\RequirePackage{csquotes} % Context-sensitive quotation marks
\RequirePackage{makeidx}  % For creating indices
\RequirePackage{xspace}   % For automatically "eating" spaces

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Multi-line comments
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\comment}[1]{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fonts & colours
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[usenames,dvipsnames]{xcolor}
\usepackage{anyfontsize}  % scales fonts if size does not exist

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Graphics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{graphicx}
\graphicspath{%
  {../Figures/}
  {./figures/}
  {./}  
}

% Suppress warnings about page groups in PDFs. This is not justified
% in most of the cases. I am pretty sure I am including my images in
% the right manner.
\begingroup\expandafter\expandafter\expandafter\endgroup
\expandafter\ifx\csname pdfsuppresswarningpagegroup\endcsname\relax
\else
  \pdfsuppresswarningpagegroup=1\relax
\fi

\RequirePackage[labelfont=bf]{caption}
\RequirePackage[]{subcaption}
\setcapindent{0pt}

% Make sub-references using \subref being typeset with parentheses.
% Otherwise, only the counter will be printed.
\captionsetup{subrefformat=parens}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Glossaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% \RequirePackage[%
%   acronym,
%   automake,
%   nogroupskip,
%   nopostdot,
%   nonumberlist,
%   toc,
%   ]{glossaries}
% 
% % New style that prevents line-breaks between the full description and
% % the acronym. Furthermore, it ensures that the acronym is always
% % printed in an upright font.
% \newacronymstyle{long-short-mimosis}
% {%
%   \GlsUseAcrEntryDispStyle{long-short}%
% }%
% {%
%   \GlsUseAcrStyleDefs{long-short}%
%   \renewcommand*{\genacrfullformat}[2]{%
%     \glsentrylong{##1}##2~\textup{(\firstacronymfont{\glsentryshort{##1}})}%
%   }%
%   \renewcommand*{\Genacrfullformat}[2]{%
%     \Glsentrylong{##1}##2~\textup{(\firstacronymfont{\glsentryshort{##1}})}%
%   }%
%   \renewcommand*{\genplacrfullformat}[2]{%
%     \glsentrylongpl{##1}##2~\textup{(\firstacronymfont{\glsentryshortpl{##1}})}%
%   }%
%   \renewcommand*{\Genplacrfullformat}[2]{%
%     \Glsentrylongpl{##1}##2~\textup{(\firstacronymfont{\Glsentryshortpl{##1}})}%
%   }%
% }
% 
% % A new glossary style that based on the long style of the glossaries
% % package. It ensures that descriptions and entries are aligned.
% \newglossarystyle{long-mimosis}{%
%   \setglossarystyle{long}
% 
%   \renewcommand{\glossentry}[2]{%
%     \glsentryitem{##1}\glstarget{##1}{\glossentryname{##1}} &
%     \glossentrydesc{##1}\glspostdescription\space ##2\tabularnewline
%   }%
% }
% 
% % Constrain the description width to enforce breaks.m
% \setlength{\glsdescwidth}{10cm}
% 
% \setacronymstyle{long-short-mimosis}
% \setglossarystyle{long-mimosis}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Paragraph lists & compact enumerations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[%
    olditem,  % Do not modify itemize environments by default
    oldenum   % Do not modify enumerate environments by default
  ]{paralist}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Spacing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{setspace}
\onehalfspacing

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{booktabs}
\RequirePackage{multirow}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Proper typesetting of units
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[binary-units=true]{siunitx}

\sisetup{%
  detect-all           = true,
  detect-family        = true,
  detect-mode          = true,
  detect-shape         = true,
  detect-weight        = true,
  detect-inline-weight = math,
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mathematics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{amsmath}
\RequirePackage{amsthm}
\RequirePackage{dsfont}

% Fix the spacing of \left and \right. Use these with the proper bracket
% in order to ensure that they scale automatically.
\let\originalleft\left
\let\originalright\right
\renewcommand{\left}{\mathopen{}\mathclose\bgroup\originalleft}
\renewcommand{\right}{\aftergroup\egroup\originalright}

\DeclareMathOperator*{\argmin}          {arg\,min}
\DeclareMathOperator {\dist}            {dist}
\DeclareMathOperator {\im}              {im}

\newcommand{\domain}{\ensuremath{\mathds{D}}}
\newcommand{\real}  {\ensuremath{\mathds{R}}}

% Proper differential operators
\newcommand{\diff}[1]{\ensuremath{\operatorname{d}\!{#1}}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ordinals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand  {\st}{\textsuperscript{\textup{st}}\xspace}
\newcommand  {\rd}{\textsuperscript{\textup{rd}}\xspace}
\newcommand  {\nd}{\textsuperscript{\textup{nd}}\xspace}
\renewcommand{\th}{\textsuperscript{\textup{th}}\xspace}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Penalties
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\clubpenalty         = 10000
\widowpenalty        = 10000
\displaywidowpenalty = 10000


% TeX prefers overlong lines to adding too much space between words on a line; 
% I think the idea is that you will notice the lines that extend into the 
% margin (and the black boxes it inserts after such lines), and will have a 
% chance to revise the contents, whereas if there was too much space, you might 
% not notice it.

% do not use sloppy!
% \sloppy nope, not a good idea

% for details, see http://texdoc.org/pkg/l2tabuen

% If line breaks appear in single paragraphs you should
%   put your text in other words. You do not necessarily need to change the sentence the line
%   break problem appears in. It may suffice to change the preceding, or the next sentence;

% Or, try these settings:

\tolerance 1414
\hbadness 1414
\emergencystretch 1.5em
\hfuzz 0.3pt
\widowpenalty=10000
\vfuzz \hfuzz
\raggedbottom

% Note that warnings appearing with the above settings really should be taken seriously. You
% should consider putting your text in other words, then.

% Check your logs for "Overfull \hbox" warnings!

% if everything else fails, use sloppypar


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Headers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% \RequirePackage{scrlayer-scrpage}
\RequirePackage[headsepline]{scrlayer-scrpage}
\pagestyle{scrheadings}
% \clearpairofpagestyles
% % \ihead{head inner}
% % \chead{head center}
% \ohead{\headmark}
% % \ifoot{foot inner}
% % \cfoot{foot center}
% \ofoot{\pagemark}

\setkomafont{pagehead}{\normalfont\selectfont\scshape}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Typefaces for parts, chapters, and sections
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand{\partformat}{\huge\partname~\thepart\autodot}
\renewcommand{\raggedpart}{\flushleft}

\setkomafont{part}{\normalfont\huge\scshape}
\setkomafont{chapter}{\normalfont\huge\sffamily\bfseries\color{BrickRed}}
% \setkomafont{chapter}{\normalfont\huge\bfseries\color{BrickRed}}
\setkomafont{sectioning}{\normalfont\sffamily\bfseries}
\setkomafont{descriptionlabel}{\normalfont\bfseries}

\setkomafont{caption}{\small}
\setkomafont{captionlabel}{\usekomafont{caption}}

% Large number for chapter
\renewcommand*{\chapterformat}{%
  \fontsize{50}{55}\selectfont\thechapter\autodot\enskip
}

\setkomafont{chapterentry}{\normalfont\large\bfseries}






% \DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrbook}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Placeholder Ducks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\usepackage{tikzducks}
\usepackage{adjustbox}
\newcommand{\exampleDuck}[1][]{%
    \adjustbox{#1}{%
      \begin{tikzpicture}
        \draw[black,fill=gray!50,thick] (0,0) rectangle (6,4);
        \draw[gray] (0,0) -- (6,4);
        \draw[gray] (0,4) -- (6,0);
        \draw[gray] (3,0) -- (3,4);
        \draw[gray] (0,2) -- (6,2);
        \node at (3,2) {\tikz{\randuck;}};
      \end{tikzpicture}
    }%
  }


