-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Mar 04, 2023 at 07:02 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `wildlife_monitor`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`username`, `password`, `mobile`) VALUES
('admin', 'admin', 0);

-- --------------------------------------------------------

--
-- Table structure for table `animal_detect`
--

CREATE TABLE `animal_detect` (
  `id` int(11) NOT NULL,
  `user` varchar(20) NOT NULL,
  `animal` varchar(20) NOT NULL,
  `image_name` varchar(40) NOT NULL,
  `dtime` timestamp NOT NULL default CURRENT_TIMESTAMP on update CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `animal_detect`
--

INSERT INTO `animal_detect` (`id`, `user`, `animal`, `image_name`, `dtime`) VALUES
(1, 'raj', 'Rhinoceros', 'c_fter43f-6.jpg', '2023-03-04 11:59:28'),
(2, 'raj', 'Elephant', 'c_23fjj-4.jpg', '2023-03-04 11:04:44'),
(3, 'raj', 'Rhinoceros', 'c_mnv4f-6.jpg', '2023-03-04 11:05:18'),
(4, 'raj', 'Bear', 'c_df434f-5.jpg', '2023-03-04 11:12:15'),
(5, 'raj', 'Lion', 'c_hj43jkf3-2.jpg', '2023-03-04 11:12:44'),
(6, '', 'Tiger', 'c_3j4j-7.jpg', '2023-03-04 11:58:25');

-- --------------------------------------------------------

--
-- Table structure for table `animal_info`
--

CREATE TABLE `animal_info` (
  `id` int(11) NOT NULL,
  `animal` varchar(30) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `animal_info`
--

INSERT INTO `animal_info` (`id`, `animal`) VALUES
(1, 'Cheetah'),
(2, 'Lion'),
(3, 'Fox'),
(4, 'Elephant'),
(5, 'Bear'),
(6, 'Rhinoceros'),
(7, 'Tiger');

-- --------------------------------------------------------

--
-- Table structure for table `register`
--

CREATE TABLE `register` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `location` varchar(50) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `pass` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `register`
--

INSERT INTO `register` (`id`, `name`, `mobile`, `email`, `location`, `uname`, `pass`) VALUES
(1, 'Raj', 8852466124, 'raj@gmail.com', 'Salem', 'raj', '123456');
