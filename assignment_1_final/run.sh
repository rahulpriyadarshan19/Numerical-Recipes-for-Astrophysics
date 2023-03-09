#!/bin/bash

echo "Run handin template"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist. Creating it..."
  mkdir plots
fi

echo "Creating the outputs directory if it does not exist"
if [ ! -d "outputs" ]; then
  echo "Directory does not exist. Creating it..."
  mkdir outputs
fi

echo "Creating the necessary output files"
echo "Creating poisson.txt"
touch outputs/poisson.txt
echo "Creating coefficients.txt"
touch outputs/coefficients.txt
echo "Creating runtimes.txt"
touch outputs/runtimes.txt

echo "Downloading the necessary data..."
if [ ! -e Vandermonde.txt ]; then
  wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/Vandermonde.txt
fi

# Script to generate Poisson distribution
echo "Running the Poisson distribution script..."
python3 poisson_distribution.py

echo "Running the Vandermonde matrix script..."
python3 vandermonde_matrix.py

echo "Generating the pdf..."

pdflatex Assignment_1.tex
bibtex Assignment_1.aux
pdflatex Assignment_1.tex
pdflatex Assignment_1.tex


