from __future__ import annotations

from ..schemas import OptionItem, QuestionDocument, SegmentationResult
from ..utils import blocks_to_segments, merge_block_texts


class SingleChoiceParser:
    def parse(self, segmentation: SegmentationResult) -> QuestionDocument:
        options = [
            OptionItem(label=label, text=merge_block_texts(blocks), bbox=blocks[0].bbox if blocks else None)
            for label, blocks in segmentation.option_groups
        ]
        stem_segments = blocks_to_segments(segmentation.stem_blocks)
        return QuestionDocument(
            question_no=segmentation.question_no,
            score=segmentation.score,
            question_type=segmentation.question_type.value,
            has_figure=bool(segmentation.figures),
            figures=segmentation.figures,
            stem_segments=stem_segments,
            stem=merge_block_texts(segmentation.stem_blocks),
            options=options,
        )
