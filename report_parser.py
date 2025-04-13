"""
MIMIC-CXR Report Parser

This module provides utilities for parsing radiology reports from the MIMIC-CXR dataset.
It includes a regex-based parser that can extract sections from reports and de-identify
sensitive information.
"""

import re
from collections import defaultdict

class MIMIC_RE:
    """
    Regular expression utility for MIMIC reports.
    Caches compiled regex patterns for better performance.
    """
    def __init__(self):
        self._cached = {}

    def get(self, pattern, flags=0):
        """Get a compiled regex pattern, using cache if available."""
        key = hash((pattern, flags))
        if key not in self._cached:
            self._cached[key] = re.compile(pattern, flags=flags)
        return self._cached[key]

    def sub(self, pattern, repl, string, flags=0):
        """Substitute pattern in string with replacement."""
        return self.get(pattern, flags=flags).sub(repl, string)

    def rm(self, pattern, string, flags=0):
        """Remove pattern from string."""
        return self.sub(pattern, '', string)

    def get_id(self, tag, flags=0):
        """Get a compiled regex pattern for an ID tag."""
        pattern = r'\[\*\*(?:{})[^\]]*\*\*\]'.format(tag)
        return self.get(pattern, flags=flags)

    def sub_id(self, tag, repl, string, flags=0):
        """Substitute ID tag in string with replacement."""
        return self.get_id(tag).sub(repl, string)

def parse_report(path):
    """
    Parse a MIMIC-CXR report from file.
    
    Args:
        path: Path to the report file
        
    Returns:
        A dictionary where keys are section titles and values are section contents
    """
    mimic_re = MIMIC_RE()
    
    # Read report file
    with open(path, 'r') as f:
        report = f.read()
    
    # Convert to lowercase
    report = report.lower()
    
    # De-identify sensitive information
    report = mimic_re.sub_id(r'(?:location|address|university|country|state|unit number)', 'LOC', report)
    report = mimic_re.sub_id(r'(?:year|month|day|date)', 'DATE', report)
    report = mimic_re.sub_id(r'(?:hospital)', 'HOSPITAL', report)
    report = mimic_re.sub_id(r'(?:identifier|serial number|medical record number|social security number|md number)', 'ID', report)
    report = mimic_re.sub_id(r'(?:age)', 'AGE', report)
    report = mimic_re.sub_id(r'(?:phone|pager number|contact info|provider number)', 'PHONE', report)
    report = mimic_re.sub_id(r'(?:name|initial|dictator|attending)', 'NAME', report)
    report = mimic_re.sub_id(r'(?:company)', 'COMPANY', report)
    report = mimic_re.sub_id(r'(?:clip number)', 'CLIP_NUM', report)

    # Handle dates in various formats
    report = mimic_re.sub(
        r'\[\*\*(?:'
            r'\d{4}'  # 1970
            r'|\d{0,2}[/-]\d{0,2}'  # 01-01
            r'|\d{0,2}[/-]\d{4}'  # 01-1970
            r'|\d{0,2}[/-]\d{0,2}[/-]\d{4}'  # 01-01-1970
            r'|\d{4}[/-]\d{0,2}[/-]\d{0,2}'  # 1970-01-01
        r')\*\*\]',
        'DATE', report
    )
    
    # Replace other patterns
    report = mimic_re.sub(r'\[\*\*[^\]]*\*\*\]', 'OTHER', report)
    report = mimic_re.sub(r'(?:\d{1,2}:\d{2})', 'TIME', report)
    
    # Clean up formatting
    report = mimic_re.rm(r'_{2,}', report, flags=re.MULTILINE)
    report = mimic_re.rm(r'the study and the report were reviewed by the staff radiologist.', report)

    # Extract sections by finding titles followed by colons
    matches = list(mimic_re.get(r'^(?P<title>[ \w()]+):', flags=re.MULTILINE).finditer(report))
    parsed_report = {}
    
    for (match, next_match) in zip(matches, matches[1:] + [None]):
        start = match.end()
        end = next_match and next_match.start()

        title = match.group('title')
        title = title.strip()

        paragraph = report[start:end]
        paragraph = mimic_re.sub(r'\s{2,}', ' ', paragraph)
        paragraph = paragraph.strip()

        parsed_report[title] = paragraph

    return parsed_report
