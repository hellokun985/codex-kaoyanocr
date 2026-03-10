from __future__ import annotations

from ..schemas import QuestionDocument, SegmentationResult, SubQuestionItem
from ..utils import blocks_to_segments, merge_block_texts


class SolutionParser:
    def parse(self, segmentation: SegmentationResult) -> QuestionDocument:
        subquestions = [
            SubQuestionItem(
                marker=marker,
                stem=merge_block_texts(blocks),
                stem_segments=blocks_to_segments(blocks),
                bbox=blocks[0].bbox if blocks else None,
            )
            for marker, blocks in segmentation.subquestion_groups
        ]
        return QuestionDocument(
            question_no=segmentation.question_no,
            score=segmentation.score,
            question_type=segmentation.question_type.value,
            has_figure=bool(segmentation.figures),
            figures=segmentation.figures,
            stem_segments=blocks_to_segments(segmentation.stem_blocks),
            stem=merge_block_texts(segmentation.stem_blocks),
            subquestions=subquestions,
        )
