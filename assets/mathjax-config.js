// MathJax 3 설정 — pymdownx.arithmatex(generic) 출력과 연동
// $...$ 인라인, $$...$$ 디스플레이 수식을 렌더링한다.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Material 인스턴트 내비게이션 후에도 수식을 다시 렌더링
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
