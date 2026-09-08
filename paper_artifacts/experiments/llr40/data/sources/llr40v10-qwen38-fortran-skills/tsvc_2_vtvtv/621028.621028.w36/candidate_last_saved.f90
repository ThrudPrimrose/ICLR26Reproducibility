subroutine benchprobe(a, b, c, len_1d) bind(C, name="tsvc_2_vtvtv_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  real(c_double) :: ts, tp, tps
  integer :: i, nproc, ios
  character(len=256) :: line, model
  real(c_double) :: total

  interface
    real(c_double) function tim_vtvtv_serial(a, b, c, n)
      use iso_c_binding
      integer(c_int64_t), intent(in) :: n
      real(c_double), intent(inout) :: a(n)
      real(c_double), intent(in) :: b(n), c(n)
    end function tim_vtvtv_serial
    real(c_double) function tim_vtvtv_par(a, b, c, n)
      use iso_c_binding
      integer(c_int64_t), intent(in) :: n
      real(c_double), intent(inout) :: a(n)
      real(c_double), intent(in) :: b(n), c(n)
    end function tim_vtvtv_par
    real(c_double) function tim_vtvtv_parsimd(a, b, c, n)
      use iso_c_binding
      integer(c_int64_t), intent(in) :: n
      real(c_double), intent(inout) :: a(n)
      real(c_double), intent(in) :: b(n), c(n)
    end function tim_vtvtv_parsimd
  end interface

  nproc = 0
  model = 'unknown'
  open(8, file='/proc/cpuinfo', status='old', action='read', iostat=ios)
  if (ios == 0) then
    do
      read(8, '(A256)', iostat=i) line
      if (i < 0) exit
      if (index(line, 'model name') > 0) model = line
      if (index(line, 'processor') > 0) nproc = nproc + 1
    end do
    close(8)
  end if

  ts  = tim_vtvtv_serial(a, b, c, len_1d)
  tp  = tim_vtvtv_par(a, b, c, len_1d)
  tps = tim_vtvtv_parsimd(a, b, c, len_1d)

  open(7, file='/shared/agent-36/bench_out.txt', form='formatted', status='replace')
  write(7, '(A)') trim(model)
  write(7, '(A,I0)') ' logical_cpus: ', nproc
  write(7, '(A,ES15.3)') ' serial     GB/s: ', 3.0d0*8.0d0*len_1d*1.0d0/(ts*1.0d9)
  write(7, '(A,ES15.3)') ' parallel   GB/s: ', 3.0d0*8.0d0*len_1d*1.0d0/(tp*1.0d9)
  write(7, '(A,ES15.3)') ' parsimd    GB/s: ', 3.0d0*8.0d0*len_1d*1.0d0/(tps*1.0d9)
  write(7, '(A,ES15.3)') ' parsimd    ms  : ', tps*1.0d3
  close(7)

  total = ts + tp + tps
  print '(A,ES15.3,A)', ' sent ', total, ' ok'
end subroutine benchprobe

real(c_double) function tim_vtvtv_serial(a, b, c, n)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), intent(in) :: n
  real(c_double), intent(inout) :: a(n)
  real(c_double), intent(in) :: b(n), c(n)
  real(c_double) :: t0, t1
  call vtvtv_serial(a, b, c, n)
  call vtvtv_serial(a, b, c, n)
  t0 = omp_get_wtime()
  call vtvtv_serial(a, b, c, n)
  t1 = omp_get_wtime()
  tim_vtvtv_serial = t1 - t0
end function tim_vtvtv_serial

subroutine vtvtv_serial(a, b, c, n)
  use iso_c_binding
  implicit none
  integer(c_int64_t), intent(in) :: n
  real(c_double), intent(inout) :: a(n)
  real(c_double), intent(in) :: b(n), c(n)
  integer(c_int64_t) :: i
  do i = 1, n
    a(i) = a(i) * b(i) * c(i)
  end do
end subroutine vtvtv_serial

real(c_double) function tim_vtvtv_par(a, b, c, n)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), intent(in) :: n
  real(c_double), intent(inout) :: a(n)
  real(c_double), intent(in) :: b(n), c(n)
  real(c_double) :: t0, t1
  call vtvtv_par(a, b, c, n)
  call vtvtv_par(a, b, c, n)
  t0 = omp_get_wtime()
  call vtvtv_par(a, b, c, n)
  t1 = omp_get_wtime()
  tim_vtvtv_par = t1 - t0
end function tim_vtvtv_par

subroutine vtvtv_par(a, b, c, n)
  use iso_c_binding
  implicit none
  integer(c_int64_t), intent(in) :: n
  real(c_double), intent(inout) :: a(n)
  real(c_double), intent(in) :: b(n), c(n)
  integer(c_int64_t) :: i
  !$omp parallel do
  do i = 1, n
    a(i) = a(i) * b(i) * c(i)
  end do
end subroutine vtvtv_par

real(c_double) function tim_vtvtv_parsimd(a, b, c, n)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), intent(in) :: n
  real(c_double), intent(inout) :: a(n)
  real(c_double), intent(in) :: b(n), c(n)
  real(c_double) :: t0, t1
  call vtvtv_parsimd(a, b, c, n)
  call vtvtv_parsimd(a, b, c, n)
  t0 = omp_get_wtime()
  call vtvtv_parsimd(a, b, c, n)
  t1 = omp_get_wtime()
  tim_vtvtv_parsimd = t1 - t0
end function tim_vtvtv_parsimd

subroutine vtvtv_parsimd(a, b, c, n)
  use iso_c_binding
  implicit none
  integer(c_int64_t), intent(in) :: n
  real(c_double), intent(inout) :: a(n)
  real(c_double), intent(in) :: b(n), c(n)
  integer(c_int64_t) :: i
  !$omp parallel do simd
  do i = 1, n
    a(i) = a(i) * b(i) * c(i)
  end do
end subroutine vtvtv_parsimd
