import * as d3 from 'd3';
import { useEffect, useRef } from 'react';

const margin = { top: 20, right: 150, bottom: 20, left: 150 };

export default function DiffBarChart({ concepts, conceptLogits, altLogits, mappedConcepts }) {
    const containerRef = useRef(null);
    const svgRef = useRef(null);
    useEffect(() => {

        if (!svgRef.current || !containerRef.current) {
            return;
        }
        const { width, height } = containerRef.current.getBoundingClientRect();
        if (width && height) {
            renderChart(svgRef.current, width, height, concepts, conceptLogits, altLogits, mappedConcepts);
        }
    }, [concepts, conceptLogits, altLogits]);
    return (
        <div ref={containerRef} style={{ width: "600px", height: "300px" }}>
            <svg
                ref={svgRef}
                width={600}
                height={300}

            ></svg>
        </div>
    );
}

function renderChart(svgElement, width, height, concepts, conceptLogits, altLogits, mappedConcepts) {
    const svg = d3.select(svgElement);
    svg.selectAll('*').remove();

    const yIndex = d3.range(concepts.length);
    const yScale = d3.scaleBand()
        .rangeRound([margin.top, height - margin.bottom])
        .domain(yIndex)
        .padding(0.1);

    const xScale = d3.scaleLinear()
        .rangeRound([margin.left, width - margin.right])
        .domain([-1.2, 1.2]);

    svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(yScale).tickFormat(i => mappedConcepts[i][0]).tickSizeOuter(0));
    svg.append('g')
        .attr('transform', `translate(${width - margin.right},0)`)
        .call(d3.axisRight(yScale).tickFormat(i => mappedConcepts[i][1]).tickSizeOuter(0));

    svg.append('g')
        .attr('transform', `translate(0,${margin.top})`)
        .call(d3.axisTop(xScale)
            .ticks(2)
            .tickFormat(d3.format('.0%')));


    const eachrow = svg.selectAll('.bar')
        .data(conceptLogits)
    eachrow
        .join('rect')
        .attr('x', (d, i) => (d - altLogits[i] >= 0 ? xScale(d) : xScale(d - altLogits[i])))
        .attr('y', (d, i) => yScale(i))
        .attr('width', (d, i) => {
            console.log("Diff width:", d - altLogits[i]);
            return d - altLogits[i] >= 0 ? (
                xScale(d - altLogits[i]) - xScale(0) - margin.left
            ) : (
                xScale(0) - xScale(d - altLogits[i])
            );
        })
        .attr('height', yScale.bandwidth())
        .attr('fill', 'pink')
        .attr('stroke', 'black');
    // eachrow
    //     .join('rect')
    //     .attr('x', d => xScale(1 - d))
    //     .attr('y', (d, i) => yScale(i))
    //     .attr('width', d => xScale(d) - margin.left)
    //     .attr('height', yScale.bandwidth())
    //     .attr('fill', 'steelblue');
}   
