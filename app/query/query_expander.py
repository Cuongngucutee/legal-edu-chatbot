"""
LawEdu AI — Education Query Expander.
Expands common education queries into legal terminology for better retrieval.
Powered by Data-Driven Domain Ontology (No hardcoded rules).
"""
import os
import json
import re

class EducationQueryExpander:
    """Expand câu hỏi giáo dục phổ thông bằng keyword mapping từ Ontology trước khi retrieve."""

    def __init__(self):
        self.doc_titles = {}
        self.expansion_rules = []
        self.target_routing_rules = []
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Load doc titles
        doc_titles_path = os.path.join(base_dir, "data", "doc_titles.json")
        if os.path.exists(doc_titles_path):
            with open(doc_titles_path, "r", encoding="utf-8") as f:
                self.doc_titles = json.load(f)
                
        # Load Domain Ontology
        ontology_path = os.path.join(base_dir, "data", "domain_knowledge.json")
        if os.path.exists(ontology_path):
            with open(ontology_path, "r", encoding="utf-8") as f:
                ontology = json.load(f).get("query_expander", {})
                self.expansion_rules = ontology.get("expansion_rules", [])
                self.target_routing_rules = ontology.get("target_routing_rules", [])
                
    def _auto_match_title(self, q: str) -> list:
        matches = []
        for so_hieu, title in self.doc_titles.items():
            t_lower = title.lower()
            t_clean = re.sub(r'^(thông tư|nghị định|luật|quyết định)\s*(hướng dẫn|quy định về việc|quy định về|quy định)?\s*', '', t_lower).strip()
            
            if not t_clean or len(t_clean) < 15:
                continue
                
            words = t_clean.split()
            stop_words = {"về", "việc", "trong", "của", "và", "các", "cho", "cơ", "sở", "giáo", "dục"}
            key_words = [w for w in words if w not in stop_words]
            
            if len(key_words) >= 4:
                matched_count = sum(1 for w in key_words if w in q)
                overlap = matched_count / len(key_words)
                
                if overlap >= 0.8 or (len(t_clean) > 20 and t_clean in q):
                    matches.append(so_hieu)
        return matches

    def expand(self, query: str) -> list:
        """Trả về list các query mở rộng dựa trên Ontology."""
        q = query.lower()
        expanded = []
        for rule in self.expansion_rules:
            if any(kw in q for kw in rule.get("keywords", [])):
                expanded.append(f"{query} {rule.get('expansion', '')}")
        return expanded

    def get_target_docs(self, query: str) -> list:
        """Trả về list các so_hieu văn bản đích dựa trên Ontology Routing Rules."""
        q = query.lower()
        targets = self._auto_match_title(q)
        
        for rule in self.target_routing_rules:
            keywords = rule.get("keywords", [])
            match_type = rule.get("match_type", "any")
            
            # All previous rules in code used `any()` implicitly
            is_match = any(kw in q for kw in keywords)
                
            if is_match:
                if "targets" in rule:
                    targets.extend(rule["targets"])
                
                if "conditions" in rule:
                    for cond in rule["conditions"]:
                        reqs = cond.get("requires", [])
                        cond_match = cond.get("match_type", "all")
                        
                        if not reqs:
                            targets.extend(cond.get("targets", []))
                        elif cond_match == "any":
                            if any(r in q for r in reqs):
                                targets.extend(cond.get("targets", []))
                        else:
                            if all(r in q for r in reqs):
                                targets.extend(cond.get("targets", []))
                                
        return list(dict.fromkeys(targets))
