# Course Recommender System

A notebook-based project exploring personalised course recommendations using course content, user profiles, enrolment/ratings data, and collaborative filtering. The repository includes exploratory analysis, text feature engineering, similarity-based recommendations, clustering, and embedding-based models.

## Project status

This is a collection of learning and experimentation notebooks, not a finished deployed application. Several notebooks contain `WRITE YOUR CODE HERE` exercises. In particular, the ANN notebook defines an embedding model but leaves its training and evaluation steps unfinished. Complete the relevant exercises before expecting a full end-to-end run.

The repaired notebook files have passed notebook-format validation and HTML conversion. This checks whether their structure can be read and rendered; it does **not** verify model correctness or successful execution of every cell. The setup below is a proposed compatibility configuration, not a fully tested environment lockfile.

## Approaches explored

- Explore course genres and user interactions.
- Represent course descriptions using tokenisation and bag-of-words features.
- Compare courses using similarity measures.
- Recommend courses from user profiles and course similarities.
- Group user profiles using K-means and explore PCA.
- Explore KNN and non-negative matrix factorisation with Surprise.
- Define a neural recommender using user and course embeddings.
- Use supplied embeddings for regression and classification experiments.

## Notebook guide and suggested order

| Order | Notebook | Purpose |
| --- | --- | --- |
| 1 | `1.1_lab_jupyter_eda.ipynb` | Exploratory data analysis |
| 2 | `1.2_lab_jupyter_fe_bow_solution.ipynb` | Text processing and bag-of-words features; includes unfinished exercises |
| 3 | `1.3_lab_jupyter_fe_course_sim.ipynb` | Course similarity feature engineering |
| 4 | `2.1_lab_jupyter_content_user_profile.ipynb` | User-profile recommendations |
| 5 | `2.2_lab_jupyter_content_course_similarity.ipynb` | Course-similarity recommendations |
| 6 | `2.3_lab_jupyter_content_clustering.ipynb` | User clustering and PCA |
| 7 | `3.1_lab_jupyter_cf_knn.ipynb` | KNN collaborative filtering |
| 8 | `3.3_lab_jupyter_cf_nmf.ipynb` | NMF collaborative filtering |
| 9 | `3.3_lab_jupyter_cf_ann.ipynb` | Neural recommender and embeddings; separate legacy setup below |
| 10 | `3.5_lab_jupyter_cf_regression_w_embeddings.ipynb` | Rating regression using supplied embeddings |
| 11 | `3.6_lab_jupyter_cf_classification_w_embeddings.ipynb` | Classification using supplied embeddings |

Additional versions are retained in `2.1_lab_jupyter_fe_course_sim.ipynb`, `3.2_lab_jupyter_cf_knn.ipynb`, `3.4_lab_jupyter_content_user_profile.ipynb`, and `module_2.1_lab_jupyter_content_user_profile.ipynb`. They are not extra mandatory pipeline stages.

`fixed.ipynb` is an earlier notebook-repair utility, not a recommendation model. It only removes certain invalid fields from text cells and does not fix every format issue; it is not needed to run the project.

## Data

The notebooks read CSV files directly from IBM Skills Network's hosted datasets. An internet connection is required for these downloads and for NLTK resources.

| Dataset | Use |
| --- | --- |
| `course_genre.csv` | Course genre features |
| `course_processed.csv` | Processed course titles and descriptions |
| `ratings.csv` | User-course interactions and ratings |
| `courses_bows.csv` | Precomputed bag-of-words features |
| `sim.csv` | Precomputed similarity matrix |
| `user_profile.csv` | User profile features |
| `user_embeddings.csv`, `course_embeddings.csv` | Precomputed embeddings |

Exact download URLs are defined inside each notebook. Some notebooks use different hosted versions of `ratings.csv`; do not assume they contain identical records. Many notebooks load precomputed features, so the order above is a learning sequence rather than a strict file-generation pipeline. The regression and classification notebooks do not require you to train the ANN first.

## Run locally: main notebooks

These instructions target Windows Command Prompt and Python 3.10 for the non-ANN notebooks. Git is optional if you download the repository ZIP instead. Python 3.10 is used here to accommodate older package pins in the existing notebooks.

### 1. Download the repository

```bash
git clone https://github.com/Jasmineqqqqq/Course-Recommender-System.git
cd Course-Recommender-System
```

Alternatively, use GitHub's **Code → Download ZIP**, extract it, and open a terminal in the extracted folder.

### 2. Create and activate an environment

In Windows Command Prompt:

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

On Linux/macOS with Python 3.10 installed, use `python3.10 -m venv .venv`, then `source .venv/bin/activate`.

### 3. Install the main dependencies

```bash
python -m pip install "numpy==1.23.5" "scipy==1.10.0" "pandas==1.5.3" "scikit-learn==1.2.2" "gensim==4.3.2" "nltk==3.6.7" "matplotlib==3.7.5" "seaborn==0.12.2" wordcloud jupyterlab ipykernel
python -m ipykernel install --user --name course-recommender --display-name "Course Recommender"
python -m pip check
```

These older versions are suggested to reduce conflicts with the notebooks' existing SciPy and NLTK installation cells. They are for reproducing these educational notebooks, not a production deployment baseline. Do not automatically upgrade individual packages without checking compatibility.

For the KNN and NMF notebooks, also install Surprise:

```bash
python -m pip install "scikit-surprise==1.1.3"
```

Surprise may need a local C/C++ compiler if a compatible prebuilt package is unavailable. On Windows, a compiler error can require Microsoft C++ Build Tools. The other notebooks can be explored without installing Surprise.

### 4. Start JupyterLab

```bash
jupyter lab
```

1. Open a notebook from the guide above.
2. Select the **Course Recommender** kernel.
3. Read the instructions and run cells from top to bottom with **Shift+Enter**.
4. Skip package-installation cells if you already installed the dependencies above; running them can change the environment. Keep the import and data-loading cells.
5. Allow the NLTK download cells to complete in the text-processing notebooks.
6. Complete any exercise placeholders before running cells that depend on their results.
7. Save the notebook after running it. Restart the kernel when switching notebooks to avoid relying on variables from a previous session.

After changing installed packages, restart the kernel and rerun the imports. Use **Run All** only after reviewing and completing the notebook's required exercises.

## ANN notebook: separate legacy environment

`3.3_lab_jupyter_cf_ann.ipynb` explicitly installs `tensorflow==2.7.0`. That release provides Python 3.7–3.9 wheels, including Windows x86-64; do not run its installation cell in the Python 3.10 environment above.

For reproducing the original ANN exercise on Windows x86-64, create a separate Python 3.9 environment:

```bat
deactivate
py -3.9 -m venv .venv-ann
.venv-ann\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install "tensorflow==2.7.0" "numpy==1.23.5" "protobuf==3.20.3" "pandas==1.5.3" "matplotlib==3.7.5" ipykernel
python -m ipykernel install --user --name course-ann-legacy --display-name "Course ANN - Legacy"
python -m pip check
```

Select **Course ANN - Legacy** in JupyterLab. JupyterLab itself can remain running from the main environment. This old Python/TensorFlow combination is only a reproduction option; upgrading the ANN to a maintained stack requires code and dependency testing beyond the notebook-format repair.

The notebook still needs its `model.compile()`, `model.fit()`, and evaluation exercises completed. Later cells such as `model.summary()` and embedding extraction depend on a built model. Installing TensorFlow alone does not complete the notebook.

## Outputs

- Tables and charts are displayed inside the notebooks.
- User-profile notebooks include an export to `profile_rs_results.csv` when the relevant cells complete.
- KNN/NMF notebooks write `course_ratings.csv` as an intermediate file for Surprise.
- Model metrics depend on completed exercises, dataset versions and train/test splits; no verified benchmark scores are claimed here.

Generated CSV files are written relative to the notebook's working directory. Re-running the export cells can overwrite files with the same names.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| GitHub shows **Invalid Notebook** | Use the latest repaired files. Format 4.4 cells must not contain a top-level `id`; Markdown cells must not contain `outputs` or `execution_count`. Preserve code-cell outputs. |
| `ModuleNotFoundError` | Confirm that the selected kernel is the environment where you installed the package. |
| `NameError` | Run preceding cells and check whether an exercise that defines the variable is unfinished. |
| NLTK `LookupError` | Run the notebook's NLTK download cells. If using a different NLTK version, follow the resource name requested by that version's error. |
| TensorFlow version not found | Use the separate legacy ANN environment or deliberately update and retest the ANN dependencies. |
| CSV download fails | Check internet access and the exact URL. If you have a local copy of the same dataset, change that notebook's `read_csv` path accordingly. |
| Packages change during execution | Skip redundant installation cells after setting up the environment, then restart the kernel. |

GitHub previews display notebooks; they do not execute Python. The repository currently has no runnable Streamlit application script, so there is no `streamlit run app.py` step.

## Acknowledgements and references

The notebooks include IBM Skills Network course material and hosted datasets. Retain the attribution and any licence notices embedded in the original notebooks; this README does not grant additional rights to third-party material.

- [JupyterLab installation documentation](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
- [TensorFlow 2.7.0 release files and supported interpreters](https://pypi.org/project/tensorflow/2.7.0/)

