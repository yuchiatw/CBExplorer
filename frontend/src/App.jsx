import { useState } from 'react'
import RenderOptions from './components/options'
import ImageDisplay from './components/image_display'
import Generator from './components/generator'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const datalist = ['cub', 'celebahq']
  const expList = ['cbae_stygan2', 'cc_stygan2']
  const [dataset, setDataset] = useState('celebahq')
  const [experiment, setExperiment] = useState('cbae_stygan2')

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
      <main>
        <div className="flex flex-row items-center justify-center p-10 gap-10">
          <Generator className="h-10" />
          <ImageDisplay className="h-10" dataset={dataset} experiment={experiment} />
        </div>
      </main>
      <footer></footer>
    </div>

  )
}

export default App
