from __future__ import annotations

from ..patterns import count_blanks
from ..schemas import BlankItem, QuestionDocument, SegmentationResult
from ..utils import blocks_to_segments, merge_block_texts


class FillBlankParser:
    def parse(self, segmentation: SegmentationResult) -> QuestionDocument:
        blanks: list[BlankItem] = []
        blank_index = 1
        for block in segmentation.blank_blocks:
            for _ in range(count_blanks(block.text)):
                blanks.append(
                    BlankItem(
                        blank_id=f"blank_{blank_index}",
                        placeholder="__",
                        bbox=block.bbox,
                    )
                )
                blank_index += 1

        return QuestionDocument(
            question_no=segmentation.question_no,
            score=segmentation.score,
            question_type=segmentation.question_type.value,
            has_figure=bool(segmentation.figures),
            figures=segmentation.figures,
            stem_segments=blocks_to_segments(segmentation.stem_blocks),
            stem=merge_block_texts(segmentation.stem_blocks),
            blanks=blanks,
        )
