@echo off

pushd %cd%
cd %~dp0


if not exist premake5.exe (
  echo premake5.exe not exists, downloading...
  echo.
  curl -L https://github.com/premake/premake-core/releases/download/v5.0.0-beta1/premake-5.0.0-beta1-windows.zip --output premake5.zip

  if not %errorlevel% == 0 (
    echo ERROR: Downloading premake5.zip failed.
    goto :End
  )

  tar -xf premake5.zip
  if not %errorlevel% == 0 (
    echo ERROR: Unzipping ^(with tar^) premake5.zip failed.
    goto :End
  )

  rm premake5.zip
  echo premake5 downloaded successfully.
  echo.
)

premake5 vs2022


:End
popd

pause
