import * as tf from "@tensorflow/tfjs";

import CSS from "../style.css";

const IMAGE_SIZE = 28;
const CANVAS_SCALE = 10;
const STROKE_WIDTH = CANVAS_SCALE * 1.5;

export default class CanvasToTensorComponent extends HTMLElement {
  private _root: ShadowRoot;
  private _canvas: HTMLCanvasElement;
  private _resultArea: HTMLDivElement;

  private _drawContext: CanvasRenderingContext2D | null = null;
  private _drawing = false;
  private _drawingStarted = false;

  private _mouseDownHandler = (e: MouseEvent) => {
    this.startDraw(e.offsetX, e.offsetY);
  };
  private _mouseMoveHandler = (e: MouseEvent) => {
    if (e.buttons) {
      this.draw(e.offsetX, e.offsetY);
    } else {
      this.endDraw();
    }
  };
  private _touchStartHandler = (e: TouchEvent) => {
    e.preventDefault();
    const touchPosition = this.getCanvasTouchPosition(e);
    this.startDraw(touchPosition.x, touchPosition.y);
  };
  private _touchMoveHandler = (e: TouchEvent) => {
    e.preventDefault();
    const touchPosition = this.getCanvasTouchPosition(e);
    this.draw(touchPosition.x, touchPosition.y);
  };
  private _touchEndHandler = (e: TouchEvent) => {
    this.endDraw();
  };

  constructor() {
    super();
    this._root = this.attachShadow({ mode: "closed" });

    // Attach styles
    const style = document.createElement("style");
    style.textContent = CSS;
    this._root.appendChild(style);

    // Create a container
    const container = document.createElement("div");
    container.className = "container";
    this._root.appendChild(container);

    // Setup canvas
    this._canvas = document.createElement("canvas");
    this._canvas.className = "draw-area";
    this._canvas.width = IMAGE_SIZE * CANVAS_SCALE;
    this._canvas.height = IMAGE_SIZE * CANVAS_SCALE;
    container.appendChild(this._canvas);

    // button
    const toTensorButton = document.createElement("button");
    toTensorButton.innerText = "to tensor";
    toTensorButton.onclick = () => {
      const imageTensor = this.createTensor();
      this.createOutputCanvas(imageTensor);
      imageTensor.dispose();

      this.clear();
      this.writeDrawHere();
    };
    container.appendChild(toTensorButton);

    // result area
    this._resultArea = document.createElement("div");
    this._resultArea.className = "result-area";
    container.appendChild(this._resultArea);
  }

  private clear() {
    if (this._drawContext) {
      this._drawContext.fillStyle = "#ffffff";
      this._drawContext.fillRect(0, 0, this._canvas.width, this._canvas.height);
    }
  }

  private startDraw(x: number, y: number) {
    if (this._drawContext) {
      if (!this._drawingStarted) {
        this._drawingStarted = true;
        this.clear();
      }
      this._drawContext.strokeRect(x, y, 1, 1);
    }
  }

  private draw(x: number, y: number) {
    if (this._drawContext) {
      if (this._drawing) {
        this._drawContext.lineTo(x, y);
        this._drawContext.stroke();
      } else {
        this._drawContext.beginPath();
        this._drawContext.moveTo(x, y);
        this._drawing = true;
      }
    }
  }

  private endDraw() {
    this._drawing = false;
  }

  private writeDrawHere() {
    this._drawingStarted = false;
    if (this._drawContext) {
      this._drawContext.fillStyle = "#0A7CC4";
      this._drawContext.font = "30px sans-serif";
      this._drawContext.textAlign = "center";
      this._drawContext.fillText(
        "Draw here!",
        this._canvas.width / 2,
        this._canvas.height / 2
      );
    }
  }

  private getCanvasTouchPosition(e: TouchEvent): { x: number; y: number } {
    const touch = e.touches[0];
    const canvasBoundingClientRect = this._canvas.getBoundingClientRect();
    return {
      x: touch.clientX - canvasBoundingClientRect.left,
      y: touch.clientY - canvasBoundingClientRect.top,
    };
  }

  /**
   * Creates a tensor from the canvas.
   */
  private createTensor() {
    return tf.tidy(() => {
      // create gauss filter for blurring
      const blurFilter = CanvasToTensorComponent.gaussFilter(
        CANVAS_SCALE,
        Math.sqrt(CANVAS_SCALE)
      );

      // combine channels -> invert -> normalize -> blur -> resize
      const image = tf.browser
        .fromPixels(this._canvas)
        .mean(2)
        .sub(255)
        .neg()
        .div(255)
        .expandDims(2)
        .conv2d(blurFilter, [1, 1], "same")
        .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
        .clipByValue(0, 1)
        .as2D(IMAGE_SIZE, IMAGE_SIZE);

      // move the center of mass into the center of the image
      const centerOfImage = tf.tensor1d([IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0]);
      const centerOfMass = CanvasToTensorComponent.centerOfMass(image);
      const distance = centerOfImage.sub(centerOfMass).as1D();
      const centeredImage = CanvasToTensorComponent.moveImage(image, distance);

      return centeredImage;
    });
  }

  /**
   * creates a canvas from the image tensor.
   * @param imageTensor image tensor
   */
  private createOutputCanvas(imageTensor: tf.Tensor2D | tf.Tensor3D) {
    const canvas = document.createElement("canvas");
    canvas.width = IMAGE_SIZE;
    canvas.height = IMAGE_SIZE;
    tf.browser.toPixels(imageTensor, canvas);
    this._resultArea.appendChild(canvas);
  }

  protected async connectedCallback() {
    if (!this.isConnected) return;
    this._drawContext = this._canvas.getContext("2d");
    if (this._drawContext) {
      this._drawContext.strokeStyle = "#000000";
      this._drawContext.lineWidth = STROKE_WIDTH;
      this._drawContext.lineCap = "round";
      this._drawContext.lineJoin = "round";

      this.clear();
      this.writeDrawHere();
    }

    this._canvas.addEventListener("mousedown", this._mouseDownHandler);
    this._canvas.addEventListener("mousemove", this._mouseMoveHandler);
    this._canvas.addEventListener("touchstart", this._touchStartHandler);
    this._canvas.addEventListener("touchmove", this._touchMoveHandler);
    this._canvas.addEventListener("touchend", this._touchEndHandler);
  }

  protected disconnectedCallback() {
    this._canvas.removeEventListener("mousedown", this._mouseDownHandler);
    this._canvas.removeEventListener("mousemove", this._mouseMoveHandler);
    this._canvas.removeEventListener("touchstart", this._touchStartHandler);
    this._canvas.removeEventListener("touchmove", this._touchMoveHandler);
    this._canvas.removeEventListener("touchend", this._touchEndHandler);
  }

  private static gaussFilter(size: number, sigma: number = 1) {
    const center = size / 2.0 - 0.5;
    const filter: number[][] = [];
    let sum = 0.0;

    // create the filter matrix
    for (let x = 0; x < size; x++) {
      filter[x] = [];
      for (let y = 0; y < size; y++) {
        const dSquared = Math.pow(x - center, 2) + Math.pow(y - center, 2);
        const exp = -dSquared / 2 / Math.pow(sigma, 2);
        sum += filter[x][y] = Math.pow(Math.E, exp);
      }
    }

    // normalize the filter!
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        filter[x][y] /= sum;
      }
    }

    return tf.tensor2d(filter, [size, size], "float32").as4D(size, size, 1, 1);
  }

  /**
   * Computes the center of mass of a greyscale image.
   */
  private static centerOfMass(image: tf.Tensor2D) {
    return tf.tidy(() => {
      const [h, w] = image.shape;
      const ii = tf.range(0, h).expandDims(1).tile([1, w]);
      const jj = tf.range(0, w).tile([h]).as2D(h, w);
      const mass = image.sum();
      const iPos = image.mul(ii).sum().div(mass);
      const jPos = image.mul(jj).sum().div(mass);
      return tf.stack([iPos, jPos]).as1D();
    });
  }

  /**
   * Moves a grayscale image by a given distance.
   */
  private static moveImage(image: tf.Tensor2D, distance: tf.Tensor1D) {
    return tf.tidy(() => {
      // gnarr...
      const d = distance.round().dataSync();
      return image
        .pad([
          [d[0], -d[0]], // top, bottom
          [d[1], -d[1]], // left, right
        ])
        .as2D(image.shape[0], image.shape[1]);
    });
  }
}
