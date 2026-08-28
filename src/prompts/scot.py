from template import Template

scot_template = Template()
scot_template.add_text(
    "You are an expert reasoner. Answer the question by reasoning step by step.", 
    mode="system")
scot_template.add_text(
    """
**CHAIN OF THOUGHT PROCESS**:
1. **ANALYZE** what the question is asking
2. **PLAN** the reasoning steps needed
3. **SOLVE** step by step
4. **STATE** the final answer clearly
    """, mode="system", ident=1)

