# Git, GitHub, and Overleaf Workflow Manual

This note documents how this directory was connected to Git and GitHub, what problems occurred, and how to repeat the workflow.

## Project Location

Local directory:

```powershell
C:\Users\juhyung\OneDrive - purdue.edu\Purdue_grad\codex_writing\Review paper\Reference\ANC_AI_ML_curated
```

GitHub repository:

```text
https://github.com/kjh44444444/review_paper_AI_in_ANC
```

## What Was Done

The local directory was initialized as a Git repository, committed, connected to GitHub, authenticated through Git Credential Manager, and pushed to GitHub.

The final pushed branch is:

```text
main
```

The latest pushed commit during setup was:

```text
e750a93 Update paper review files
```

## One-Time Setup Steps

Use these steps only when starting a new local folder that is not yet connected to Git.

1. Open PowerShell in the project directory.

```powershell
cd "C:\Users\juhyung\OneDrive - purdue.edu\Purdue_grad\codex_writing\Review paper\Reference\ANC_AI_ML_curated"
```

2. Initialize Git.

```powershell
git init
```

3. Check the repository state.

```powershell
git status
```

4. Stage all files.

```powershell
git add .
```

5. Create the first commit.

```powershell
git commit -m "Initial paper project"
```

6. Rename the default branch to `main`.

```powershell
git branch -M main
```

7. Connect the local repository to GitHub.

```powershell
git remote add origin https://github.com/kjh44444444/review_paper_AI_in_ANC
```

8. Push the local branch to GitHub.

```powershell
git push -u origin main
```

After this step, GitHub should show the files from the local directory.

## Normal Daily Workflow

Use these steps after editing files.

1. Check what changed.

```powershell
git status
```

2. Stage all changed files.

```powershell
git add .
```

Or stage one file:

```powershell
git add paper_reviews.md
```

3. Commit the staged changes.

```powershell
git commit -m "Describe what changed"
```

Example:

```powershell
git commit -m "Update paper review files"
```

4. Push to GitHub.

```powershell
git push
```

5. Confirm the repository on GitHub.

```text
https://github.com/kjh44444444/review_paper_AI_in_ANC
```

## Connecting GitHub to Overleaf

Use this after the GitHub repository has the files.

1. Log in to Overleaf.
2. Go to Account Settings.
3. Link the GitHub account.
4. Create a new Overleaf project.
5. Choose Import from GitHub.
6. Select:

```text
kjh44444444/review_paper_AI_in_ANC
```

7. Open the imported project in Overleaf.

Important: Overleaf GitHub sync is manual. After pushing changes from the local computer to GitHub, use Overleaf's GitHub sync option to pull the changes into Overleaf.

## Problems Encountered and Fixes

### Problem 1: Files Did Not Appear on GitHub

The files were committed locally, but GitHub did not show them.

Diagnosis:

```powershell
git log --oneline -5
git remote -v
git branch -vv
```

The local commit existed, and `origin` pointed to GitHub, but the branch had not been pushed successfully.

Fix:

```powershell
git push -u origin main
```

### Problem 2: GitHub Authentication Failed

GitHub rejected the push with:

```text
Invalid username or token. Password authentication is not supported for Git operations.
```

Cause:

GitHub no longer accepts account passwords for Git pushes over HTTPS. Git needs a valid token stored through Git Credential Manager.

Fix:

```powershell
git credential-manager github logout https://github.com
git credential-manager github login
git push -u origin main
```

After logging in again, the push succeeded.

### Problem 3: `git add .` Failed With `index.lock` Permission Denied

Git failed with:

```text
fatal: Unable to create '.git/index.lock': Permission denied
```

Cause:

Git needed to write to its internal index file. The first attempt was blocked by file permissions/sandbox behavior.

Fix:

Run `git add` again from a normal PowerShell session with permission to write to the `.git` directory.

```powershell
git add .
```

### Problem 4: Word Document Could Not Be Staged

Git failed with:

```text
error: open("ANC_AI_ML_trend_analysis_and_paper_reviews.docx"): Permission denied
error: unable to index file 'ANC_AI_ML_trend_analysis_and_paper_reviews.docx'
fatal: updating files failed
```

Cause:

The Word document was likely open in Microsoft Word, locked by OneDrive sync, or temporarily unavailable.

Fix:

Close the Word document, wait for OneDrive to finish syncing, then stage the file again:

```powershell
git add ANC_AI_ML_trend_analysis_and_paper_reviews.docx
```

Then verify:

```powershell
git status --short
```

### Problem 5: Local Branch Was Ahead of GitHub

After committing, Git showed:

```text
main...origin/main [ahead 1]
```

Cause:

The commit existed locally but had not been pushed to GitHub yet.

Fix:

```powershell
git push
```

## Useful Status Commands

Check changed files:

```powershell
git status
```

Compact status:

```powershell
git status --short --branch
```

Check recent commits:

```powershell
git log --oneline -5
```

Check GitHub remote:

```powershell
git remote -v
```

Check whether the local branch tracks GitHub:

```powershell
git branch -vv
```

## Practical Notes

Avoid editing the same file in Word, Overleaf, and a local editor at the same time. This can create file locks or sync conflicts.

For LaTeX journal writing, Git works best with text files such as `.tex`, `.bib`, `.md`, and `.csv`. Word `.docx` files can be stored in Git, but Git cannot show meaningful line-by-line changes inside them.

Recommended paper structure:

```text
main.tex
sections/
  01_introduction.tex
  02_methods.tex
  03_results.tex
  04_discussion.tex
figures/
tables/
references.bib
```

In `main.tex`, include section files like this:

```latex
\input{sections/01_introduction}
\input{sections/02_methods}
\input{sections/03_results}
\input{sections/04_discussion}
```

