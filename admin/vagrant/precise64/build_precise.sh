#!/bin/bash

post="_precise";
branch="branch-2-05";
release="$HOME/release";
#---------------------------------------------------------------------------------------
builder="precise64_build.sh";
builder_adr="../../builders/"
precise_box_url="http://files.vagrantup.com/precise64.box";
#---------------------------------------------------------------------------------------
#managing source and release directories
if [ $1 ]
then tmp_dir=$1;
else tmp_dir="$(mktemp -d --tmpdir precise_tmp_XXXX)";
mkdir ${tmp_dir}/src;
git clone --branch ${branch} https://github.com/percolator/percolator.git ${tmp_dir}/src/percolator;
fi;
if [ $2 ]
then release=$2;
fi;
#---------------------------------------------------------------------------------------
# making directories and copy builder:
mkdir -p ${release};
cp ${builder_adr}${builder} ${tmp_dir};
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
# making the Vagrantfile:
cd ${tmp_dir};
touch Vagrantfile;
#-----------------Vagrantfile content---------------
cat <<EOF > Vagrantfile
# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "precise64"
  config.vm.box_url = "${precise_box_url}"
  config.vm.provision :shell, :inline => "su vagrant -c 'bash /vagrant/${builder} /vagrant/src /vagrant/build${post}'"
end
EOF
#-----------------end of Vagrantfile content--------

#---------------------------------------------------------------------------------------
vagrant up

#---------------------------------------------------------------------------------------
# release:
cp ${tmp_dir}/build${post}/percolator/per*.rpm ${release};
cp ${tmp_dir}/build${post}/percolator/per*.deb ${release};
cp ${tmp_dir}/build${post}/converters/per*.rpm ${release};
cp ${tmp_dir}/build${post}/converters/per*.deb ${release};

#---------------------------------------------------------------------------------------
vagrant destroy -f
#---------------------------------------------------------------------------------------
cd ${release};
