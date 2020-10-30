import * as tf from "@tensorflow/tfjs";

import CSS from "../style.css";

const IMAGE_SIZE = 28;
const CANVAS_SCALE = 10;
const STROKE_WIDTH = CANVAS_SCALE * 1.6;

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
    const button = document.createElement("button");
    button.innerText = "to tensor";
    button.onclick = () => {
      this.createTensor();
      this.clear();
      this.writeDrawHere();
    };
    container.appendChild(button);

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

  private createTensor() {
    tf.tidy(() => {
      // create tensor from drawn image
      const imageTensor = tf.browser
        .fromPixels(this._canvas)
        .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])
        .mean(2)
        .sub(255)
        .neg()
        .div(255)
        .as2D(IMAGE_SIZE, IMAGE_SIZE);

      // create image canvas from tensor
      const canvas = document.createElement("canvas");
      canvas.width = IMAGE_SIZE;
      canvas.height = IMAGE_SIZE;
      tf.browser.toPixels(imageTensor, canvas);
      this._resultArea.appendChild(canvas);
    });
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
}
