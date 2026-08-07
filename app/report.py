import os
from fpdf import FPDF
from pathlib import Path
from datetime import datetime

class DiagnosticReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(33, 37, 41)
        self.cell(0, 10, "CHEST X-RAY CLINICAL DECISION SUPPORT SYSTEM", border=False, ln=True, align="C")
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(108, 117, 125)
        self.cell(0, 6, "Explainable AI Diagnostic Assessment & Referral Report", border=False, ln=True, align="C")
        self.line(10, 28, 200, 28)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(108, 117, 125)
        self.cell(0, 10, "CONFIDENTIAL - For Clinical Decision Support Only. Not a standalone medical diagnosis.", border=False, align="C")

def generate_pdf_report(patient_name, patient_id, pred_label, confidence, prob_dict, orig_img_path, gradcam_img_path, output_path="results/report.pdf"):
    pdf = DiagnosticReportPDF()
    pdf.add_page()

    # Patient & Assessment Info Table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Patient & Case Metadata", ln=True)
    pdf.set_font("Helvetica", "", 10)
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(95, 6, f"Patient Name: {patient_name}", border=True)
    pdf.cell(95, 6, f"Patient ID: {patient_id}", border=True, ln=True)
    pdf.cell(95, 6, f"Scan Timestamp: {current_time}", border=True)
    pdf.cell(95, 6, f"Primary Diagnosis: {pred_label.upper()}", border=True, ln=True)
    pdf.cell(95, 6, f"Model Confidence: {confidence*100:.1f}%", border=True)
    
    # Triage Level
    if pred_label in ["tuberculosis", "bacterial_pneumonia"]:
        triage = "RED ALERT (High Suspicion - Immediate Referral)"
    elif pred_label == "viral_pneumonia":
        triage = "YELLOW ALERT (Moderate Suspicion - Secondary Monitoring)"
    else:
        triage = "GREEN (Low Risk / Normal Scan)"
        
    pdf.cell(95, 6, f"Triage Status: {triage}", border=True, ln=True)
    pdf.ln(5)

    # Class Probability Breakdown
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. Differential Probability Distribution", ln=True)
    pdf.set_font("Helvetica", "", 10)
    
    for cls_name, prob in prob_dict.items():
        pdf.cell(70, 6, f"  • {cls_name.replace('_', ' ').title()}:", border=False)
        pdf.cell(40, 6, f"{prob*100:.2f}%", border=False, ln=True)

    pdf.ln(5)

    # Visual Evidence & Heatmap Section
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3. Visual Evidence & Grad-CAM Heatmap Localization", ln=True)
    pdf.ln(2)

    # Embed images side-by-side if paths exist
    if os.path.exists(orig_img_path) and os.path.exists(gradcam_img_path):
        y_pos = pdf.get_y()
        pdf.image(orig_img_path, x=15, y=y_pos, w=80)
        pdf.image(gradcam_img_path, x=105, y=y_pos, w=80)
        pdf.set_y(y_pos + 85)

    # Physician Sign-off Box
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Attending Physician Sign-off:", ln=True)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 5, "Signature: ___________________________    Date: ______________", ln=True)

    # Output file
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf.output(output_path)
    return output_path
