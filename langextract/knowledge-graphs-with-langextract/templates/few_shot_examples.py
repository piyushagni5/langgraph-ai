import langextract as lx

def get_dynamic_examples(query: str) -> list:
    """Return appropriate few-shot examples based on query keywords"""
    
    if any(keyword in query.lower() for keyword in ['financial', 'revenue', 'company']):
        # Business-focused examples
        return [
            lx.data.ExampleData(
                text="Apple Inc. reported $394.3 billion in revenue for fiscal 2022. The company is headquartered in Cupertino, California.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="company",
                        extraction_text="Apple Inc.",
                        attributes={"type": "technology_company"}
                    ),
                    lx.data.Extraction(
                        extraction_class="financial_metric", 
                        extraction_text="$394.3 billion in revenue",
                        attributes={"period": "fiscal 2022", "metric_type": "revenue"}
                    ),
                    lx.data.Extraction(
                        extraction_class="location",
                        extraction_text="Cupertino, California",
                        attributes={"type": "headquarters"}
                    ),
                ]
            )
        ]
    elif any(keyword in query.lower() for keyword in ['legal', 'contract']):
        # Legal-focused examples
        return [
            lx.data.ExampleData(
                text="The agreement between XYZ Corp and ABC Ltd was signed on January 15, 2024, with a term of 5 years.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="party",
                        extraction_text="XYZ Corp",
                        attributes={"role": "contractor"}
                    ),
                    lx.data.Extraction(
                        extraction_class="party",
                        extraction_text="ABC Ltd", 
                        attributes={"role": "client"}
                    ),
                    lx.data.Extraction(
                        extraction_class="date",
                        extraction_text="January 15, 2024",
                        attributes={"event": "signing_date"}
                    ),
                ]
            )
        ]
    else:
        # Generic example
        return [
            lx.data.ExampleData(
                text="Romeo loved Juliet deeply, despite their families' feud in Verona.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="character",
                        extraction_text="Romeo",
                        attributes={"role": "protagonist"}
                    ),
                    lx.data.Extraction(
                        extraction_class="character",
                        extraction_text="Juliet",
                        attributes={"role": "protagonist"}
                    ),
                    lx.data.Extraction(
                        extraction_class="location",
                        extraction_text="Verona",
                        attributes={"type": "city"}
                    ),
                ]
            )
        ]