subroutine tsvc_2_s316_fp64(a, result, n) bind(C, name="tsvc_2_s316_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), dimension(:), intent(in)  :: a
  real(c_double), intent(out)               :: result
  integer(c_int64_t)                        :: n
  integer(c_int64_t) :: i
  real(c_double) :: x
  logical, save :: printed = .false.
  if (.not. printed) then
    printed = .true.
    call write_i("PROBE n=", n)
    call write_i("PROBE maxthreads=", int(omp_get_max_threads(),8))
    call dump_cpuinfo()
    call flush()
  end if
  x = a(1)
  do i = 2, n
    if (a(i) < x) x = a(i)
  end do
  result = x
end subroutine

subroutine write_i(tag, val)
  use iso_c_binding
  implicit none
  character(len=*), intent(in) :: tag
  integer(c_int64_t), intent(in) :: val
  write(*,*) tag, val
end subroutine

subroutine dump_cpuinfo()
  implicit none
  character(len=4096) :: line
  integer :: iu, ios
  logical :: seen_model, seen_flags, seen_sib, seen_cores, seen_numa
  seen_model=.false.; seen_flags=.false.; seen_sib=.false.; seen_cores=.false.; seen_numa=.false.
  open(newunit=iu, file='/proc/cpuinfo', status='old', action='read', iostat=ios)
  if (ios /= 0) then
    write(*,*) 'PROBE cpuinfo open fail'
    return
  end if
  do
    read(iu, '(A)', iostat=ios) line
    if (ios /= 0) exit
    if (.not. seen_model .and. index(line, 'model name') > 0) then
      seen_model=.true.; write(*,*) 'CPUINFO ', trim(line)
    end if
    if (.not. seen_flags .and. index(line, 'flags') > 0) then
      seen_flags=.true.; write(*,*) 'CPUINFO ', trim(line)
    end if
    if (.not. seen_sib .and. index(line, 'siblings') > 0) then
      seen_sib=.true.; write(*,*) 'CPUINFO ', trim(line)
    end if
    if (.not. seen_cores .and. index(line, 'cpu cores') > 0) then
      seen_cores=.true.; write(*,*) 'CPUINFO ', trim(line)
    end if
    if (.not. seen_numa .and. index(line, 'numa_node') > 0) then
      seen_numa=.true.; write(*,*) 'CPUINFO ', trim(line)
    end if
  end do
  close(iu)
end subroutine
