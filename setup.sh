# this script assumes that you are running on Ubuntu 18.04 and have sudo rights

# install dependencies
echo "Installing m4"
sudo apt-get install m4

echo "Installing gmp"
#sudo apt-get install -y libgmp-dev
wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
tar -xvf gmp-6.1.2.tar.xz
cd gmp-6.1.2
./configure --enable-cxx
make
sudo make install
cd ..
rm gmp-6.1.2.tar.xz

echo "Installing mpfr"
#sudo apt-get install libmpfr-dev libmpfr-doc
wget https://files.sri.inf.ethz.ch/eran/mpfr/mpfr-4.1.0.tar.xz
tar -xvf mpfr-4.1.0.tar.xz
cd mpfr-4.1.0
./configure || true
make
sudo make install
cd ..
rm mpfr-4.1.0.tar.xz

echo "Installing cddlib"
wget https://github.com/cddlib/cddlib/releases/download/0.94m/cddlib-0.94m.tar.gz
tar zxf cddlib-0.94m.tar.gz
rm cddlib-0.94m.tar.gz
cd cddlib-0.94m
./configure
make
sudo make install
cd ..

  
# download repo
# git clone https://gitlab.inf.ethz.ch/markmueller/prima4complete.git
# cd prima4complete

echo "Installing ELINA"
# setup ELINA
git clone https://github.com/eth-sri/ELINA.git
cd ELINA
./configure -use-deeppoly -use-fconv
make
sudo make install
cd ..

echo "Installing GUROBI"
wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz
tar -xvf gurobi9.1.2_linux64.tar.gz
cd gurobi912/linux64/src/build
sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile
make
cp libgurobi_c++.a ../../lib/
cd ../../
sudo cp lib/libgurobi91.so /usr/local/lib
cd ../../
rm gurobi9.1.2_linux64.tar.gz

export GUROBI_HOME="$(pwd)/gurobi912/linux64"
export PATH="${PATH}:/usr/lib:${GUROBI_HOME}/bin"
export CPATH="${CPATH}:${GUROBI_HOME}/include"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib:${GUROBI_HOME}/lib

echo $(ls)