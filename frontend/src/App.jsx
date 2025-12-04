import { useState, useEffect } from 'react'
import RenderOptions from './components/options'
import ImageDisplay from './components/IntialImage'
import Generator from './components/generator'
import ManipulateImage from './components/manipulate_image'
import './App.css'

function App() {
  const datalist = ['cub', 'celebahq']
  const expList = ['cbae_stygan2', 'cc_stygan2']
  const [dataset, setDataset] = useState('celebahq')
  const [experiment, setExperiment] = useState('cbae_stygan2')
  const [concepts, setConcepts] = useState([]);
  const [seed, setSeed] = useState(60);

  const [originalImageSrc, setOriginalImageSrc] = useState('');
  const [conceptLogits, setConceptLogits] = useState([]);
  const [bits, setBits] = useState([...new Array(8).fill(0)]);

  useEffect(() => {
    fetch("http://localhost:8000/concepts/" + experiment + "/" + dataset)
      .then(response => response.json())
      .then(data => {
        console.log("Fetched concepts:", data);
        if (data.concepts) {
          setConcepts(data.concepts);
        }
      })
      .catch(error => {
        console.error("Error fetching concepts:", error);
      });
    fetch("http://localhost:8000/generate/" + experiment + "/" + dataset + "/" + seed)
      .then(response => response.json())
      .then(data => {
        console.log("Image generated with seed:", seed);
        if (data.image) {
          setOriginalImageSrc(data.image);
        }
        if (data.concept_probs) {
          setConceptLogits(data.concept_probs);
        }
        if (data.concept_values) {
          setBits(data.concept_values);
          console.log("Initial concept values:", data.concept_values);
        }

      })
      .catch(error => {
        console.error("Error generating image:", error);
      });
  }, [dataset, experiment]);

  return (
    <div className="flex flex-col min-h-screen">
      <header className="flex flex-row header-container bg-zinc-900 text-center text-white py-4 px-8 text-xl">
        <h2 className="text-left text-2xl">CBExplorer</h2>
        <label htmlFor="data-select" className="mx-2">
          <select
            id='data-select'
            className="bg-gray-200 text-black p-2 rounded mx-2"
            value={dataset}
            onChange={e => setDataset(e.target.value)}
          >
            {datalist.map((data, index) => (
              <option
                key={index}
                value={data}
              >
                {data}
              </option>
            ))}
          </select>
        </label>
        <label htmlFor="experiment-select" className="mx-2">
          <select
            id='experiment-select'
            className="bg-gray-200 text-black p-2 rounded mx-2"
            value={experiment}
            onChange={e => setExperiment(e.target.value)}
          >
            {expList.map((exp, index) => (
              <option
                key={index}
                value={exp}
              >
                {exp}
              </option>
            ))}
          </select>
        </label>
      </header>
      <main className="flex flex-col items-center flex-grow">
        <div className="flex flex-row items-center justify-center p-10 gap-10">
          <Generator className="h-10" />
          <ImageDisplay
            className="h-10"
            concepts={concepts}
            imageSrc={originalImageSrc}
            conceptLogits={conceptLogits}
          />
        </div>
        <div>
          <ManipulateImage
            className="h-10"
            dataset={dataset}
            experiment={experiment}
            concepts={concepts}
            bits={bits}
            setBits={setBits}
            seed={seed}
          />
        </div>
      </main>
      <footer></footer>
    </div>

  )
}

export default App
