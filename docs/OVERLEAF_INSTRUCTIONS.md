# How to Use the LaTeX File in Overleaf

## Quick Steps

1. **Upload the LaTeX file:**
   - Go to your Overleaf project
   - Upload `docs/individual-progress-report.tex`

2. **Upload the figures:**
   - In Overleaf, create a folder structure: `reports/figures/`
   - Upload these three images to that folder:
     - `reports/figures/confusion_matrix_test.png`
     - `reports/figures/confusion_matrix_test_normalized.png`
     - `reports/figures/roc_curves_test.png`

3. **Compile:**
   - Set the compiler to **pdfLaTeX** (Overleaf default)
   - Click "Recompile"
   - Your PDF should generate successfully!

## Files to Upload to Overleaf

From your project directory:
```
docs/individual-progress-report.tex          → Main document (root of Overleaf project)
reports/figures/confusion_matrix_test.png    → Upload to reports/figures/ folder
reports/figures/confusion_matrix_test_normalized.png → Upload to reports/figures/ folder
reports/figures/roc_curves_test.png          → Upload to reports/figures/ folder
```

## Alternative: Use Online Image URLs

If you don't want to upload the images, you can modify the LaTeX file to use direct GitHub URLs instead:

Replace the image paths at the end of the document from:
```latex
\includegraphics{../reports/figures/confusion_matrix_test.png}
```

To:
```latex
\includegraphics{https://raw.githubusercontent.com/hmolhem/nthu-driver-drowsiness-ROI/main/reports/figures/confusion_matrix_test.png}
```

Note: This requires setting `\usepackage{graphicx}` with shell-escape or using `\href` for links instead.

## Expected Output

The PDF will include:
- Professional formatting with sections and subsections
- Tables (overfitting analysis, training time, test results, per-class metrics)
- Three evaluation figures (confusion matrices and ROC curves)
- GitHub repository links
- Code snippets in monospace font
