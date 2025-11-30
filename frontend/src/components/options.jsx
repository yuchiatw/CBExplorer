const list = ['Celeb', 'Animals']
export default function RenderOptions() {
    const bars = list;
    return bars.map((bar, index) => (
        <option key={index} value={bar}>{bar}</option>
    ));
}