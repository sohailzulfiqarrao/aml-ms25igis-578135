# Applied Machine Learning, Fall 2026

Assignment work for the Applied Machine Learning course, MS Remote Sensing and GIS, Institute of Geographical
Information Systems (IGIS), NUST. One repository for the whole semester, one folder per assignment, published as a
single GitHub Pages site.

**Site:** https://sohailzulfiqarrao.github.io/aml-ms25igis-578135/

## Assignments

| Assignment | Folder | Results page | Report | Notebook |
|---|---|---|---|---|
| 01 – Predicting Evapotranspiration over Islamabad | [`Assignments/Assignment 1/`](Assignments/Assignment%201/) | [open](https://sohailzulfiqarrao.github.io/aml-ms25igis-578135/Assignments/Assignment%201/) | [report](Assignments/Assignment%201/report.md) | [notebook](Assignments/Assignment%201/Islamabd_MLR_Assignment.ipynb) |

## Layout

```
index.html                   landing page listing every assignment
assets/site.css              its stylesheet, shared by every page
Assignments/Assignment 1/    Assignment 01: multiple linear regression
    index.html  report.md  images/  Islamabd_MLR_Assignment.ipynb  Islamabad_Data.csv
```

Each new assignment gets its own folder under `Assignments/` and a new card on the landing page.

## Python on GitHub

Git stores `.py` files and Jupyter notebooks (`.ipynb`) like any other text file, and GitHub shows both: `.py` with
syntax highlighting, notebooks rendered with their code, output and plots. GitHub Pages only serves static files, so
it cannot execute Python; the site links to the rendered notebook and to Google Colab for running it.

## Run it locally

To preview the site, do not open a page by double-clicking it. Serve the repository folder instead, for example with
the VS Code **Live Server** extension (right-click `index.html`, *Open with Live Server*, which opens
`http://127.0.0.1:5500`), or from a terminal:

```
python -m http.server 5500
```

Then open:

| Page | Local address |
|---|---|
| Landing page | `http://127.0.0.1:5500/` |
| Assignment 01 | `http://127.0.0.1:5500/Assignments/Assignment%201/` |

To run the notebook, use Jupyter from the `Assignments/Assignment 1` folder, where it finds `Islamabad_Data.csv`:

```
pip install notebook pandas scikit-learn matplotlib seaborn statsmodels tabulate
jupyter notebook
```

Or open it in [Google Colab](https://colab.research.google.com/github/sohailzulfiqarrao/aml-ms25igis-578135/blob/main/Assignments/Assignment%201/Islamabd_MLR_Assignment.ipynb)
and upload the CSV to the session first.

## Publish updates

Updates are pushed with a local `commit.bat` that stages everything, commits as "Updating coursework" and pushes. It
is listed in `.gitignore`, so it stays on the local machine. Lecture slides, the in-class notebooks and other course
files are kept out of the repository by `.gitignore` too.

---
**Author:** Sohail Zulfiqar, MS GIS & Remote Sensing
