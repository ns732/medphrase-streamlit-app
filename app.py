        st.caption("BioBERT predicts adverse-reaction spans and their character offsets.")
        st.markdown("**02 — Represent**")
        st.caption("SapBERT encodes each extracted phrase using CLS pooling.")
        st.markdown("**03 — Rank**")
        st.caption("Cosine similarity orders candidate SNOMED CT concepts from most to less similar.")
    st.markdown(
        '<div class="note"><b>Interpretation note.</b> Similarity is a ranking signal, '
        'not a calibrated probability. Suggestions require human review and are not diagnoses.</div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-kicker">Analysis / 03</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Model suggestions</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">Predictions are shown as returned by the models. '
    'Incorrect spans and imperfect concept matches are useful evidence for evaluation.</div>',
    unsafe_allow_html=True,
)

if "analysis_results" not in st.session_state:
    st.markdown(
        '<div class="empty-state"><div style="font-size:25px;margin-bottom:8px;">⌁</div>'
        '<b>Your results will appear here.</b><br>Run an analysis to view extracted phrases, '
        'character offsets, and ranked terminology suggestions.</div>',
        unsafe_allow_html=True,
    )
else:
    results = st.session_state["analysis_results"]
    source_text = st.session_state.get("analysis_source_text", "")
    candidate_count = st.session_state.get("analysis_candidate_count", 0)
    metrics = st.columns(3, gap="medium")
    with metrics[0]:
        render_metric("Candidate spans", str(len(results)), "Predicted by BioBERT")
    with metrics[1]:
        render_metric("Concept vocabulary", f"{candidate_count:,}", "Candidate concepts")
    with metrics[2]:
        render_metric("Ranked candidates", str(sum(len(x["ranked_concepts"]) for x in results)), "Up to five per phrase")

    st.markdown("#### Extracted phrases & top suggestions")
    if not results:
        st.warning("BioBERT did not identify any ADR spans in this text. This can be a model error; it does not prove that no side effect is present.")
    else:
        for index, result in enumerate(results, start=1):
            ranked = result["ranked_concepts"]
            top = ranked[0] if ranked else None
            if top:
                st.markdown(
                    f'<div class="result-card"><div class="result-top">'
                    f'<div><div class="mention">{index:02d} &nbsp; {html.escape(result["mention"])}</div>'
                    f'<div class="offset">Characters {result["start"]}–{result["end"]}</div></div>'
                    f'<div class="sim-badge">Cosine {top["cosine_similarity"]:.3f}</div></div>'
                    f'<div class="concept-label">Top-ranked concept</div>'
                    f'<div class="concept-name">{html.escape(top["snomed_name"])}</div>'
                    f'<div class="concept-id">SNOMED CT · {html.escape(top["snomed_id"])}</div></div>',
                    unsafe_allow_html=True,
                )
            with st.expander(f"View top-five ranking for “{result['mention']}”"):
                rank_df = pd.DataFrame([
                    {
                        "Rank": i,
                        "SNOMED CT ID": concept["snomed_id"],
                        "Concept name": concept["snomed_name"],
                        "Cosine similarity": round(concept["cosine_similarity"], 4),
                    }
                    for i, concept in enumerate(ranked, start=1)
                ])
                st.dataframe(rank_df, hide_index=True, use_container_width=True)

        with st.expander("View input text and predicted spans"):
            span_df = pd.DataFrame([
                {"Phrase": x["mention"], "Start": x["start"], "End": x["end"],
                 "Original text check": source_text[x["start"]:x["end"]]}
                for x in results
            ])
            st.dataframe(span_df, hide_index=True, use_container_width=True)
            export_rows = []
            for result in results:
                for rank, concept in enumerate(result["ranked_concepts"], start=1):
                    export_rows.append({
                        "mention": result["mention"],
                        "start": result["start"],
                        "end": result["end"],
                        "rank": rank,
                        "snomed_id": concept["snomed_id"],
                        "snomed_name": concept["snomed_name"],
                        "cosine_similarity": concept["cosine_similarity"],
                    })
            export_df = pd.DataFrame(export_rows)
            if not export_df.empty:
                st.download_button(
                    "Download results as CSV",
                    data=export_df.to_csv(index=False).encode("utf-8"),
                    file_name="medphrase_patient_side_effect_analysis.csv",
                    mime="text/csv",
                )

st.markdown(
    '<div class="footer"><span>MEDPHRASE · MAAI 902 NLP SOLUTION DESIGN</span>'
    '<span>RESEARCH DEMO ONLY · NOT FOR CLINICAL USE</span></div>',
    unsafe_allow_html=True,
)
