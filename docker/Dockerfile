FROM nvcr.io/nvidia/pytorch:21.07-py3
# configure sshd settings
ENV TZ=Asia/Tokyo
RUN mkdir ~/.ssh
ADD ./authorized_keys /root/.ssh/
RUN mkdir /var/run/sshd
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update
RUN apt install -y openssh-server
RUN echo "Port 22" >> /etc/ssh/sshd_config
EXPOSE 22
ADD ./packages.txt /packages.txt
ADD ./requirements.txt /requirements.txt
ADD ./ipaexg.ttf /ipaexg.ttf
RUN apt install -y $(grep -vE "^\s*#" /packages.txt  | tr "\n" " ")
RUN pip3 install -r /requirements.txt
ADD ./cmd.sh /cmd.sh
ADD ./wandb.sh /wandb.sh
ENTRYPOINT []
WORKDIR /mnt
