#!/usr/bin/bash

pandoc body/*.md --pdf-engine=pdflatex -o final.tex --template template.tex  --filter pandoc-secnos --filter pandoc-eqnos --filter pandoc-tablenos --filter pandoc-fignos --filter pandoc-crossref --citeproc --csl ../ieee.csl -M cref=false --standalone

inotifywait -q -m -e close_write -r "body" | 
while read -r filename event; do
    pandoc body/*.md --pdf-engine=pdflatex -o final.pdf --template template.tex  --filter pandoc-secnos --filter pandoc-eqnos --filter pandoc-tablenos --filter pandoc-fignos --filter pandoc-crossref --citeproc --csl ../ieee.csl -M cref=false --standalone
done

# pandoc main.md --pdf-engine=pdflatex -o final.tex --filter pandoc-secnos --filter pandoc-eqnos --filter pandoc-tablenos --filter pandoc-fignos --filter pandoc-crossref --citeproc --csl ../ieee.csl -M cref=false