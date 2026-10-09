"""Create the first-MVP submission sheet; requires ReportLab only for authoring."""
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'pdf' / 'Context_Surgeon_MVP_Submission.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
INK = colors.HexColor('#17342E')
MUTED = colors.HexColor('#52665F')
GREEN = colors.HexColor('#16735C')
PALE = colors.HexColor('#EEF5F0')
LINE = colors.HexColor('#D3E1D8')
styles = {
    'body': ParagraphStyle('Body', fontName='Helvetica', fontSize=9.2, leading=13.4, textColor=INK, spaceAfter=5),
    'small': ParagraphStyle('Small', fontName='Helvetica', fontSize=8, leading=11.2, textColor=MUTED),
    'label': ParagraphStyle('Label', fontName='Helvetica-Bold', fontSize=8.2, leading=12, textColor=GREEN, spaceBefore=12, spaceAfter=5),
    'metric': ParagraphStyle('Metric', fontName='Helvetica-Bold', fontSize=19, leading=24, textColor=GREEN),
    'title': ParagraphStyle('Title', fontName='Helvetica-Bold', fontSize=31, leading=36, textColor=INK, spaceAfter=4),
    'subtitle': ParagraphStyle('Subtitle', fontName='Helvetica', fontSize=11, leading=16, textColor=MUTED, spaceAfter=12),
}
def p(text, style='body'):
    return Paragraph(text, styles[style])
story = [p('MITK AI VISION 24H  /  PS-02 - COMMVAULT CHALLENGE', 'label'),
         p('Context Surgeon', 'title'),
         p('Explainable context compression for LLM requests<br/><b>First MVP submission</b>  |  09 October 2026', 'subtitle')]
story += [p('01  PROTOTYPE DESCRIPTION', 'label'),
          p('A lightweight local API and browser interface that accept a system prompt, reference context and user query, then return a shorter, query-relevant context. The MVP aims to reduce unnecessary input tokens while retaining the evidence needed to answer. Compression runs locally without an extra LLM call.')]
story += [p('02  KEY FEATURES &amp; FUNCTIONALITY', 'label'),
          p('<b>Explainable compression.</b> Splits content into blocks, scores relevance, removes exact duplicates and explains why each block was kept or removed.'),
          p('<b>Evidence protection.</b> Preserves system instructions, corrections, constraints, recent blocks, linked identifiers and Python dependencies. A soft budget allows essential evidence to exceed the target.'),
          p('<b>Evaluation workflow.</b> Includes 40 labeled synthetic cases across conversations, documents, code and structured data; local benchmarking; and downloadable JSON reports.'),
          p('<b>Automatic Gemini selection.</b> Finds a compatible model, switches on model availability or quota errors, and uses the same model for both answers in each comparison. Successful API stages are cached for resume.'),
          p('<b>Tools:</b> Python 3.10+ standard library (HTTP server, AST, JSON, urllib); HTML, CSS and JavaScript; Google Gemini API for answer comparisons and actual token counts. No third-party packages are required to run the app.')]
summary = json.loads((ROOT / 'results' / 'offline-benchmark.json').read_text(encoding='utf-8'))['summary']
metrics = Table([[
    [p(str(summary['cases']), 'metric'), p('Synthetic benchmark cases', 'small')],
    [p(f"{summary['mean_estimated_reduction_pct']:.2f}%", 'metric'), p('Mean estimated token reduction', 'small')],
    [p(f"{summary['all_facts_retained_cases']}/{summary['cases']}", 'metric'), p('Cases retaining labeled evidence', 'small')],
]], colWidths=[166,166,167])
metrics.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),('BOX',(0,0),(-1,-1),.5,LINE),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)]))
story += [Spacer(1,6), metrics, Spacer(1,5), p('Local counts estimate UTF-8 bytes / 4. Labeled-evidence retention is not answer correctness; manual review of generated answers remains pending.', 'small')]
story += [p('03  TEAM DETAILS', 'label')]
team = Table([[p('<b>Saad</b>'), p('<b>Tarun</b>'), p('<b>Gagan</b>')],
              [p('Dataset preparation', 'small'), p('Ideation', 'small'), p('Design', 'small')]], colWidths=[166,166,167])
team.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),1)]))
story += [team, p('04  PROTOTYPE DEMONSTRATION', 'label'),
          p('<b>Local demo:</b> <link href="http://127.0.0.1:8000" color="#16735C">http://127.0.0.1:8000</link> &nbsp; | &nbsp; Start with <font name="Courier">python app.py</font>'),
          p('Select a sample or paste input, run <b>Compress locally</b>, inspect section decisions, then run the offline benchmark or <b>Compare this example</b>. The server chooses the Gemini model automatically.'),
          p('The demo URL works on the computer running the app; this MVP has no public deployment. Gemini evaluation requires an API key and available quota.', 'small'),
          p('05  MVP SCOPE &amp; NEXT STEPS', 'label'),
          p('This is the first working MVP. Windows file-save retries and recovery preserve completed evaluation stages. Further work includes broader robustness testing, semantic relevance, larger datasets and formal human scoring of answer quality.', 'small')]

def page(canvas, doc):
    canvas.setTitle('Context Surgeon - First MVP Submission')
    canvas.setAuthor('Saad, Tarun, Gagan')
    canvas.setStrokeColor(LINE)
    canvas.line(48, 39, A4[0]-48, 39)
    canvas.setFillColor(MUTED)
    canvas.setFont('Helvetica',7.5)
    canvas.drawString(48,26,'CONTEXT SURGEON  |  AI/ML - PERFORMANCE OPTIMIZATION')
    canvas.drawRightString(A4[0]-48,26,f'{doc.page}')

SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=48, leftMargin=48,
                  topMargin=25, bottomMargin=51).build(story, onFirstPage=page, onLaterPages=page)
print(OUT)
