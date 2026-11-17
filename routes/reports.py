"""
Report Generation and Export Routes (CSV & PDF).
"""

import io
import csv
import logging
from datetime import datetime
from flask import Blueprint, render_template, request, make_response, jsonify

from database.db import (
    get_analysis_history,
    get_batch_job,
    get_batch_results,
    get_dashboard_stats,
    get_model_metrics,
    get_all_batch_jobs
)

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

logger = logging.getLogger(__name__)
reports_bp = Blueprint("reports", __name__)


@reports_bp.route("/reports", methods=["GET"])
def reports_page():
    """Render reports and export management interface."""
    stats = get_dashboard_stats()
    metrics = get_model_metrics()
    batches = get_all_batch_jobs(limit=20)
    return render_template("reports.html", stats=stats, metrics=metrics, batches=batches)


@reports_bp.route("/reports/csv", methods=["GET"])
def export_csv():
    """Export analysis records or specific batch to CSV."""
    batch_id = request.args.get("batch_id")

    si = io.StringIO()
    writer = csv.writer(si)

    if batch_id:
        try:
            bid = int(batch_id)
            batch = get_batch_job(bid)
            results = get_batch_results(bid)
            filename = f"batch_{bid}_sentiment_export.csv"

            writer.writerow(["ID", "Batch_ID", "Input_Text", "Predicted_Sentiment", "Positive_Score", "Negative_Score", "Neutral_Score"])
            for r in results:
                writer.writerow([
                    r["id"],
                    r["batch_id"],
                    r["input_text"],
                    r["sentiment"],
                    r.get("positive_score", ""),
                    r.get("negative_score", ""),
                    r.get("neutral_score", "")
                ])
        except Exception as e:
            logger.exception("Failed to export batch CSV")
            return jsonify({"status": "error", "message": str(e)}), 400
    else:
        # Export latest analysis history (up to 1000 records)
        records, _ = get_analysis_history(page=1, per_page=1000)
        filename = f"sentiment_analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        writer.writerow(["ID", "Input_Text", "Sentiment", "Model_Used", "Positive_Score", "Negative_Score", "Neutral_Score", "Type", "Created_At"])
        for r in records:
            writer.writerow([
                r["id"],
                r["input_text"],
                r["sentiment"],
                r["model_name"],
                r.get("positive_score", ""),
                r.get("negative_score", ""),
                r.get("neutral_score", ""),
                r.get("analysis_type", ""),
                r.get("created_at", "")
            ])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename={filename}"
    output.headers["Content-Type"] = "text/csv; charset=utf-8"
    return output


@reports_bp.route("/reports/pdf", methods=["GET"])
def export_pdf():
    """
    Generate and stream an analytical PDF summary report using ReportLab.
    """
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        # Custom purple-accent styles
        primary_purple = colors.HexColor("#7C3AED")
        dark_purple = colors.HexColor("#4C1D95")
        light_bg = colors.HexColor("#F8F7FC")
        text_dark = colors.HexColor("#1F2937")

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=primary_purple,
            fontName="Helvetica-Bold",
            spaceAfter=6
        )

        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#6B7280"),
            spaceAfter=15
        )

        section_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontSize=13,
            leading=17,
            textColor=dark_purple,
            fontName="Helvetica-Bold",
            spaceBefore=12,
            spaceAfter=8
        )

        cell_style = ParagraphStyle(
            "CellNormal",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=text_dark
        )

        cell_bold = ParagraphStyle(
            "CellBold",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.white,
            fontName="Helvetica-Bold"
        )

        elements = []

        # Header
        elements.append(Paragraph("Sentiment Analysis Analytics & Model Performance Report", title_style))
        elements.append(Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')} | System: College Major Project Specification",
            subtitle_style
        ))
        elements.append(Spacer(1, 10))

        # Section 1: Live Sentiment Summary
        elements.append(Paragraph("1. System Sentiment Summary", section_style))
        stats = get_dashboard_stats()
        total = stats["total_analyzed"]
        pos = stats["positive_count"]
        neg = stats["negative_count"]
        neu = stats["neutral_count"]

        pos_pct = f"{(pos / total * 100):.1f}%" if total > 0 else "0.0%"
        neg_pct = f"{(neg / total * 100):.1f}%" if total > 0 else "0.0%"
        neu_pct = f"{(neu / total * 100):.1f}%" if total > 0 else "0.0%"

        summary_data = [
            [Paragraph("Metric", cell_bold), Paragraph("Count", cell_bold), Paragraph("Percentage", cell_bold)],
            [Paragraph("Total Texts Analyzed", cell_style), Paragraph(str(total), cell_style), Paragraph("100.0%", cell_style)],
            [Paragraph("Positive Sentiment", cell_style), Paragraph(str(pos), cell_style), Paragraph(pos_pct, cell_style)],
            [Paragraph("Negative Sentiment", cell_style), Paragraph(str(neg), cell_style), Paragraph(neg_pct, cell_style)],
            [Paragraph("Neutral Sentiment", cell_style), Paragraph(str(neu), cell_style), Paragraph(neu_pct, cell_style)],
            [Paragraph("Total Bulk Batches", cell_style), Paragraph(str(stats["total_batches"]), cell_style), Paragraph("-", cell_style)],
        ]

        summary_table = Table(summary_data, colWidths=[240, 150, 150])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), primary_purple),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 15))

        # Section 2: Model Performance Benchmark
        elements.append(Paragraph("2. Supervised Learning Algorithms Comparison (Held-Out Test Data)", section_style))
        metrics = get_model_metrics()

        model_data = [
            [
                Paragraph("Algorithm", cell_bold),
                Paragraph("Accuracy", cell_bold),
                Paragraph("Macro Precision", cell_bold),
                Paragraph("Macro Recall", cell_bold),
                Paragraph("Macro F1", cell_bold),
                Paragraph("Test Samples", cell_bold)
            ]
        ]

        for m in metrics:
            model_data.append([
                Paragraph(m["model_name"], cell_style),
                Paragraph(f"{m['accuracy'] * 100:.2f}%", cell_style),
                Paragraph(f"{m['precision'] * 100:.2f}%", cell_style),
                Paragraph(f"{m['recall'] * 100:.2f}%", cell_style),
                Paragraph(f"{m['f1_score']:.4f}", cell_style),
                Paragraph(str(m["testing_samples"]), cell_style),
            ])

        model_table = Table(model_data, colWidths=[150, 80, 80, 80, 75, 75])
        model_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), dark_purple),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(model_table)
        elements.append(Spacer(1, 15))

        # Section 3: Recent Activity Sample
        elements.append(Paragraph("3. Recent Analysis Activity", section_style))
        records, _ = get_analysis_history(page=1, per_page=5)

        history_data = [
            [
                Paragraph("ID", cell_bold),
                Paragraph("Sample Input Text", cell_bold),
                Paragraph("Sentiment", cell_bold),
                Paragraph("Model", cell_bold),
                Paragraph("Timestamp", cell_bold)
            ]
        ]

        for r in records:
            truncated = (r["input_text"][:50] + "...") if len(r["input_text"]) > 50 else r["input_text"]
            history_data.append([
                Paragraph(str(r["id"]), cell_style),
                Paragraph(truncated, cell_style),
                Paragraph(r["sentiment"], cell_style),
                Paragraph(r["model_name"], cell_style),
                Paragraph(str(r["created_at"])[:16], cell_style)
            ])

        if len(records) == 0:
            history_data.append([
                Paragraph("-", cell_style),
                Paragraph("No records found.", cell_style),
                Paragraph("-", cell_style),
                Paragraph("-", cell_style),
                Paragraph("-", cell_style)
            ])

        history_table = Table(history_data, colWidths=[35, 230, 85, 110, 80])
        history_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), primary_purple),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(history_table)

        # Build PDF document
        doc.build(elements)
        buffer.seek(0)

        response = make_response(buffer.getvalue())
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = f"attachment; filename=sentiment_analytics_report_{datetime.now().strftime('%Y%m%d')}.pdf"
        return response

    except Exception as e:
        logger.exception("Failed to generate PDF report")
        return jsonify({"status": "error", "message": f"Failed to generate PDF report: {str(e)}"}), 500
