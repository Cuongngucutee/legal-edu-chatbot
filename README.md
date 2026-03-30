# Legal Knowledge Graph Full

Workspace nay da duoc toi gian de tap trung vao:
- Xay dung `knowledge_graph_full.json` tu du lieu trong `data/final`
- Test truy van/relevance tren do thi da build
- Quan sat ket qua GraphRAG bang UI

## Cau truc hien tai

- `data/final`: du lieu luat JSON dau vao (giu nguyen)
- `scripts/build_knowledge_graph.py`: build KG full
- `scripts/test_graphrag_kg_full.py`: test truy van tren KG full
- `scripts/graphrag_ui.py`: UI quan sat truy xuat hybrid
- `outputs/knowledge_graph/knowledge_graph_full.json`: file KG output

## Chay nhanh

Khong can thu vien ngoai cho 2 script hien tai (chi dung Python standard library).

```bash
python scripts/build_knowledge_graph.py
python scripts/test_graphrag_kg_full.py
```

## Chay UI de quan sat GraphRAG

```bash
pip install -r requirements.txt
streamlit run scripts/graphrag_ui.py
```

## Benchmark va cham diem tu dong

- Bo benchmark mac dinh: `benchmarks/kg_query_benchmark.json`
- Bo benchmark sua doi/bai bo: `benchmarks/graphrag_amend_repeal_benchmark.json`
- Script danh gia: `scripts/evaluate_kg_benchmark.py`

Chay danh gia tong the:

```bash
python scripts/evaluate_kg_benchmark.py
```

Danh gia bo benchmark sua doi/bai bo:

```bash
python scripts/evaluate_kg_benchmark.py --benchmark benchmarks/graphrag_amend_repeal_benchmark.json --top-k 7
```

Chi hien thi case fail:

```bash
python scripts/evaluate_kg_benchmark.py --fail-only
```

Xuat report JSON de theo doi qua cac lan tinh chinh:

```bash
python scripts/evaluate_kg_benchmark.py --json-output outputs/benchmark/benchmark_report.json
```

Test voi cau hoi tuy chinh:

```bash
python scripts/test_graphrag_kg_full.py --question "Ai co tham quyen quy dinh chi tiet ve van bang, chung chi?" --top-k 5
```

In ket qua JSON:

```bash
python scripts/test_graphrag_kg_full.py --json
```