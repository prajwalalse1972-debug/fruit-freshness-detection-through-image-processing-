from fpdf import FPDF
import csv
import os
from datetime import datetime

def generate_pdf(csv_file="predictions.csv", output_file="Freshness_Report.pdf"):
    if not os.path.exists(csv_file):
        return False, "No predictions.csv found. Make some predictions first!"

    rows = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return False, "No predictions found in CSV!"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    # Header
    pdf.set_fill_color(21, 101, 192)
    pdf.rect(0, 0, 210, 38, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_y(8)
    pdf.cell(0, 10, "Fruit Freshness Detection Report",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, "A Smartphone-Based System Using Image Processing",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Generated: {datetime.now().strftime('%d %B %Y  %H:%M')}",
             align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)

    # Summary stats
    total  = len(rows)
    fresh  = sum(1 for r in rows if r["Result"] == "FRESH")
    rotten = total - fresh
    fruits = {}
    for r in rows:
        fruits[r["Fruit Type"]] = fruits.get(r["Fruit Type"], 0) + 1

    pdf.set_text_color(10, 22, 40)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)

    pdf.set_fill_color(240, 248, 255)
    pdf.cell(0, 7, f"  Total Predictions : {total}",
             new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.cell(0, 7, f"  Fresh             : {fresh}  ({round(fresh/total*100,1)}%)",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"  Rotten            : {rotten}  ({round(rotten/total*100,1)}%)",
             new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.cell(0, 7, f"  Fruits Scanned    : {', '.join(fruits.keys())}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Table header
    pdf.set_fill_color(21, 101, 192)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    col_w = [42, 28, 55, 28, 32]
    headers = ["Timestamp", "Fruit", "Image File", "Result", "Confidence"]
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 8, h, border=1, fill=True, align="C")
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", "", 9)
    for idx, row in enumerate(rows):
        if idx % 2 == 0:
            pdf.set_fill_color(240, 248, 255)
        else:
            pdf.set_fill_color(255, 255, 255)

        pdf.set_text_color(50, 50, 50)
        pdf.cell(col_w[0], 7, row["Timestamp"], border=1, fill=True)
        pdf.cell(col_w[1], 7, row["Fruit Type"], border=1, fill=True, align="C")
        pdf.cell(col_w[2], 7, row["Image File"][:28], border=1, fill=True)

        # Colour result cell
        if row["Result"] == "FRESH":
            pdf.set_fill_color(200, 240, 215)
            pdf.set_text_color(0, 100, 60)
        else:
            pdf.set_fill_color(255, 220, 220)
            pdf.set_text_color(180, 30, 30)

        pdf.cell(col_w[3], 7, row["Result"], border=1, fill=True, align="C")

        # Reset for confidence
        if idx % 2 == 0:
            pdf.set_fill_color(240, 248, 255)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(col_w[4], 7, f"{row['Confidence %']}%",
                 border=1, fill=True, align="C")
        pdf.ln()

    # Footer
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6,
             "Dayananda Sagar College of Engineering  |  Dept. of Electronics & Telecommunication  |  1BPRJ208",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6,
             "Team: Prajwal Alse - Pavan Kumar KS - Pari Singh - Pranav Puranik  |  Guide: Dr. Priya",
             align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.output(output_file)
    return True, output_file


if __name__ == "__main__":
    success, msg = generate_pdf()
    if success:
        print(f"✓ PDF saved as: {msg}")
    else:
        print(f"✗ Error: {msg}")
