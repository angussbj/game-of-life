const GRID_SIZE = 512;

const canvas = document.getElementById("canvas");
if (!navigator.gpu) {
  throw new Error("WebGPU not supported on this browser.");
}
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
  throw new Error("No appropriate GPUAdapter found.");
}
const device = await adapter.requestDevice();
const context = canvas.getContext("webgpu");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format: canvasFormat });

const vertices = new Float32Array([
  -0.7, -0.7, 0.7, -0.7, 0.7, 0.7, -0.7, -0.7, 0.7, 0.7, -0.7, 0.7,
]);
const vertexBuffer = device.createBuffer({
  label: "Cell vertices",
  size: vertices.byteLength,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, vertices);
const vertexBufferLayout = {
  arrayStride: 8,
  attributes: [
    {
      format: "float32x2",
      offset: 0,
      shaderLocation: 0,
    },
  ],
};

const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
const uniformBuffer = device.createBuffer({
  label: "Grid uniforms",
  size: uniformArray.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

const cellStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);
const cellStateStorage = [
  device.createBuffer({
    label: "Cell state A",
    size: cellStateArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  }),
  device.createBuffer({
    label: "Cell state B",
    size: cellStateArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  }),
];
for (let i = 0; i < cellStateArray.length; i++) {
  const random = Math.random();
  cellStateArray[i] =
    random < 0.04
      ? 255
      : random < 0.08
      ? 255 << 8
      : random < 0.12
      ? 255 << 16
      : 0;
}
console.log((255 << 8).toString(2));
device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);

const cellShaderModule = device.createShaderModule({
  label: "Cell shader",
  code: `
    struct VertexInput {
      @location(0) pos: vec2f,
      @builtin(instance_index) instance: u32,
    }

    struct VertexOutput {
      @builtin(position) pos: vec4f,
      @location(0) cell: vec2f,
    }

    @group(0) @binding(0) var<uniform> grid: vec2f;
    @group(0) @binding(1) var<storage> cellState: array<u32>;

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
      let i = f32(input.instance);
      let cell = vec2f(i % grid.x, floor(i / grid.x));
      let cellOffset = cell * 2 / grid;
      let gridPos = (input.pos + 1) / grid - 1 + cellOffset;
      let state = f32(cellState[input.instance]);

      var output: VertexOutput;
      output.pos = vec4f(gridPos, 0, 1) * state;
      output.cell = cell;
      return output;
    }

    fn cellIndex(cell: vec2f) -> u32 {
      return u32((cell.y % grid.y) * grid.x + (cell.x % grid.x));
    }

    @fragment
    fn fragmentMain(@location(0) cell: vec2f) -> @location(0) vec4f {
      let color = cellState[cellIndex(cell)];
      return vec4f(f32(color & 255) / 255, f32((color & (255 << 8)) >> 8) / 255, f32((color & (255 << 16)) >> 16) / 255, 1);
    }
  `,
});

const WORKGROUP_SIZE = 8;
const simulationShaderModule = device.createShaderModule({
  label: "Game of life simulation shader",
  code: `
    @group(0) @binding(0) var<uniform> grid: vec2f;
    @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
    @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

    fn cellIndex(cell: vec2u) -> u32 {
      return (cell.y % u32(grid.y)) * u32(grid.x) + (cell.x % u32(grid.x));
    }

    fn cellColor(x: u32, y: u32) -> u32 {
      return cellStateIn[cellIndex(vec2(x, y))];
    }

    fn averageColor(colors: array<u32, 8>) -> u32 {
      var r: u32 = 0;
      var g: u32 = 0;
      var b: u32 = 0;
      for (var i = 0; i < 8; i++) {
        r = r + (colors[i] & 255);
        g = g + (colors[i] & (255 << 8));
        b = b + (colors[i] & (255 << 16));
      }
      return (
        (((r + 2) / 3) & 255) + 
        (((g + (2 << 8)) / 3) & (255 << 8)) + 
        (((b + (2 << 16)) / 3) & (255 << 16))
      );
    }

    fn countNonZero(colors: array<u32, 8>) -> u32 {
      var count: u32 = 0;
      for (var i = 0; i < 8; i++) {
        if (colors[i] != 0) {
          count = count + 1;
        }
      }
      return count;
    }

    @compute
    @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
    fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
      let neighbourColors = array<u32, 8>(cellColor(cell.x+1, cell.y+1),
                        cellColor(cell.x+1, cell.y),
                        cellColor(cell.x+1, cell.y-1),
                        cellColor(cell.x, cell.y-1),
                        cellColor(cell.x-1, cell.y-1),
                        cellColor(cell.x-1, cell.y),
                        cellColor(cell.x-1, cell.y+1),
                        cellColor(cell.x, cell.y+1));
      let i = cellIndex(cell.xy);  
      switch countNonZero(neighbourColors) {
        case 2: {
          cellStateOut[i] = cellStateIn[i];
        }
        case 3: {
          cellStateOut[i] = averageColor(neighbourColors);
        }
        default: {
          cellStateOut[i] = 0;
        }
      }
    }
  `,
});

const bindGroupLayout = device.createBindGroupLayout({
  label: "Cell bind group layout",
  entries: [
    {
      binding: 0,
      visibility:
        GPUShaderStage.VERTEX |
        GPUShaderStage.COMPUTE |
        GPUShaderStage.FRAGMENT,
      buffer: {}, // grid uniform
    },
    {
      binding: 1,
      visibility:
        GPUShaderStage.VERTEX |
        GPUShaderStage.COMPUTE |
        GPUShaderStage.FRAGMENT,
      buffer: { type: "read-only-storage" }, // cell state input
    },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" }, // cell state output
    },
  ],
});

const bindGroups = [
  device.createBindGroup({
    label: "Cell renderer bind group A",
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: { buffer: uniformBuffer },
      },
      {
        binding: 1,
        resource: { buffer: cellStateStorage[0] },
      },
      {
        binding: 2,
        resource: { buffer: cellStateStorage[1] },
      },
    ],
  }),
  device.createBindGroup({
    label: "Cell renderer bind group B",
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: { buffer: uniformBuffer },
      },
      {
        binding: 1,
        resource: { buffer: cellStateStorage[1] },
      },
      {
        binding: 2,
        resource: { buffer: cellStateStorage[0] },
      },
    ],
  }),
];

const pipelineLayout = device.createPipelineLayout({
  label: "Cell Pipeline Layout",
  bindGroupLayouts: [bindGroupLayout],
});

const cellPipeline = device.createRenderPipeline({
  label: "Cell pipeline",
  layout: pipelineLayout,
  vertex: {
    module: cellShaderModule,
    entryPoint: "vertexMain",
    buffers: [vertexBufferLayout],
  },
  fragment: {
    module: cellShaderModule,
    entryPoint: "fragmentMain",
    targets: [
      {
        format: canvasFormat,
      },
    ],
  },
});

const simulationPipeline = device.createComputePipeline({
  label: "Simulation pipeline",
  layout: pipelineLayout,
  compute: {
    module: simulationShaderModule,
    entryPoint: "computeMain",
  },
});

const UPDATE_INTERVAL = 200;
let step = 0;

function updateGrid() {
  const encoder = device.createCommandEncoder();
  const computePass = encoder.beginComputePass();
  computePass.setPipeline(simulationPipeline);
  computePass.setBindGroup(0, bindGroups[step % 2]);
  const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
  computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
  computePass.end();

  step++;

  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        storeOp: "store",
        // clearValue: { r: 0.1, g: 0.2, b: 0.3, a: 1 },
      },
    ],
  });

  pass.setPipeline(cellPipeline);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.setBindGroup(0, bindGroups[step % 2]);
  pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE);
  pass.end();
  device.queue.submit([encoder.finish()]);

  window.requestAnimationFrame(updateGrid);
}

updateGrid();
