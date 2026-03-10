from __future__ import annotations

from .data_models import BBox, LineCluster, OCRBlock


class YLineClusterer:
    def cluster(self, blocks: list[OCRBlock]) -> list[LineCluster]:
        text_blocks = [block for block in blocks if block.text.strip()]
        if not text_blocks:
            return []

        if all(block.line_id for block in text_blocks):
            grouped: dict[str, list[OCRBlock]] = {}
            for block in text_blocks:
                grouped.setdefault(block.line_id, []).append(block)
            items = list(grouped.items())
        else:
            items = self._cluster_by_center_y(text_blocks)

        clusters: list[LineCluster] = []
        for reading_order, (line_id, line_blocks) in enumerate(items, start=1):
            sorted_blocks = sorted(line_blocks, key=lambda block: (block.bbox.x1, block.block_id))
            bbox = BBox(
                x1=min(block.bbox.x1 for block in sorted_blocks),
                y1=min(block.bbox.y1 for block in sorted_blocks),
                x2=max(block.bbox.x2 for block in sorted_blocks),
                y2=max(block.bbox.y2 for block in sorted_blocks),
            )
            clusters.append(
                LineCluster(
                    line_id=line_id,
                    bbox=bbox,
                    blocks=sorted_blocks,
                    text=" ".join(block.text for block in sorted_blocks if block.text).strip(),
                    reading_order=reading_order,
                )
            )
        return sorted(clusters, key=lambda line: (line.reading_order, line.bbox.y1, line.bbox.x1))

    def _cluster_by_center_y(self, blocks: list[OCRBlock]) -> list[tuple[str, list[OCRBlock]]]:
        sorted_blocks = sorted(blocks, key=lambda block: (block.bbox.y1, block.bbox.x1, block.block_id))
        avg_height = sum(max(1, block.bbox.height) for block in sorted_blocks) / len(sorted_blocks)
        threshold = max(10, int(avg_height * 0.6))

        groups: list[list[OCRBlock]] = []
        centers: list[float] = []
        for block in sorted_blocks:
            center_y = (block.bbox.y1 + block.bbox.y2) / 2
            if not groups:
                groups.append([block])
                centers.append(center_y)
                continue

            if abs(center_y - centers[-1]) <= threshold:
                groups[-1].append(block)
                centers[-1] = (centers[-1] * (len(groups[-1]) - 1) + center_y) / len(groups[-1])
            else:
                groups.append([block])
                centers.append(center_y)

        return [(f"line_{index}", group) for index, group in enumerate(groups, start=1)]
