import * as d3 from 'd3';
import { useEffect, useRef } from 'react';

const margin = { top: 20, right: 30, bottom: 40, left: 70 };
const data = [0.5, 0.6, 0.8, 0.4, 0.9, 0.7, 0.3, 0.2, 1.0, 0.55];

export default function BarChart({ concepts, conceptLogits }) {
    const containerRef = useRef(null);
    const svgRef = useRef(null);
    useEffect(() => {

        if (!svgRef.current || !containerRef.current) {
            return;
        }
        const { width, height } = containerRef.current.getBoundingClientRect();
        if (width && height) {
            renderChart(svgRef.current, width, height, concepts, conceptLogits);
        }
    }, [concepts, conceptLogits]);
    return (
        <div ref={containerRef} style={{ width: "400px", height: "400px" }}>
            <svg
                ref={svgRef}
                width={400}
                height={400}
            ></svg>
        </div>
    );
}

function renderChart(svgElement, width, height, concepts, conceptLogits) {
    const svg = d3.select(svgElement);
    svg.selectAll('*').remove();

    const yIndex = d3.range(concepts.length);
    const yScale = d3.scaleBand()
        .rangeRound([margin.top, height - margin.bottom])
        .domain(yIndex)
        .padding(0.1);

    const xScale = d3.scaleLinear()
        .rangeRound([margin.left, width - margin.right])
        .domain([0, 1]);

    svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(yScale).tickFormat(i => concepts[i]).tickSizeOuter(0));

    svg.append('g')
        .attr('transform', `translate(0,${margin.top})`)
        .call(d3.axisTop(xScale).ticks(10, '%'));

    const eachrow = svg.selectAll('.bar')
        .data(conceptLogits)
    eachrow
        .join('rect')
        .attr('x', xScale(0))
        .attr('y', (d, i) => yScale(i))
        .attr('width', (d) => xScale(d) - margin.left)
        .attr('height', yScale.bandwidth())
        .attr('fill', 'steelblue');
    eachrow
        .join('rect')
        .attr('x', d => xScale(d))
        .attr('y', (d, i) => yScale(i))
        .attr('width', d => xScale(1 - d) - margin.left)
        .attr('height', yScale.bandwidth())
        .attr('fill', 'pink');
}   
