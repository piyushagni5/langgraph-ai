import langextract as lx

def get_dynamic_examples(query: str) -> list:
    """Return appropriate few-shot examples based on query keywords"""
    
    if any(keyword in query.lower() for keyword in ['financial', 'revenue', 'company', 'business', 'founder', 'ceo']):
        # Business-focused examples with relationships
        return [
            lx.data.ExampleData(
                text="Apple Inc. reported $394.3 billion in revenue for fiscal 2022. The company is headquartered in Cupertino, California. Steve Jobs founded Apple Inc.",
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
                    lx.data.Extraction(
                        extraction_class="person",
                        extraction_text="Steve Jobs",
                        attributes={"role": "founder"}
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="Apple Inc. is headquartered in Cupertino, California",
                        attributes={"type": "located_in", "subject": "Apple Inc.", "object": "Cupertino, California"}
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="Steve Jobs founded Apple Inc.",
                        attributes={"type": "founded_by", "subject": "Steve Jobs", "object": "Apple Inc."}
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
                    lx.data.Extraction(
                        extraction_class="term",
                        extraction_text="5 years",
                        attributes={"duration": "5 years"}
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="XYZ Corp signed agreement with ABC Ltd",
                        attributes={"type": "contractual_agreement", "subject": "XYZ Corp", "object": "ABC Ltd"}
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="Agreement was signed on January 15, 2024",
                        attributes={"type": "signed_on", "subject": "agreement", "object": "January 15, 2024"}
                    ),
                ]
            )
        ]
    else:
        # Generic example with relationships - diverse domain
        return [
            lx.data.ExampleData(
                text="Romeo loved Juliet deeply, despite their families' feud in Verona. They met at the Capulet party.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="character",
                        extraction_text="Romeo",
                        attributes={"role": "protagonist", "family": "Montague"}
                    ),
                    lx.data.Extraction(
                        extraction_class="character",
                        extraction_text="Juliet",
                        attributes={"role": "protagonist", "family": "Capulet"}
                    ),
                    lx.data.Extraction(
                        extraction_class="location",
                        extraction_text="Verona",
                        attributes={"type": "city"}
                    ),
                    lx.data.Extraction(
                        extraction_class="event",
                        extraction_text="Capulet party",
                        attributes={"type": "social_gathering"}
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="Romeo loved Juliet",
                        attributes={"type": "romantic_love", "subject": "Romeo", "object": "Juliet"}
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="Romeo and Juliet met at the Capulet party",
                        attributes={"type": "met_at", "subject": "Romeo and Juliet", "object": "Capulet party"}
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="families' feud in Verona",
                        attributes={"type": "conflict", "subject": "Montague and Capulet families", "object": "Verona"}
                    ),
                ]
            )
        ]