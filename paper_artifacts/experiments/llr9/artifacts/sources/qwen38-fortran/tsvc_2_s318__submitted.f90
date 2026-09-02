subroutine tsvc_2_s318_fp64(a_ptr, result_ptr, len_1d, inc, workspace, ws_bytes) bind(C, name="tsvc_2_s318_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  type(c_ptr), intent(in), value :: a_ptr
  type(c_ptr), intent(in), value :: result_ptr
  integer(c_int64_t), intent(in), value :: len_1d
  integer(c_int64_t), intent(in), value :: inc
  type(c_ptr), intent(in), value :: workspace
  integer(c_int64_t), intent(in), value :: ws_bytes

  real(c_double), dimension(:), pointer :: a
  real(c_double), dimension(:), pointer :: result
  integer(c_int64_t) :: i, nt, tid, start, end_
  real(c_double) :: maxv, v, gmaxv
  integer(c_int64_t) :: gindex
  real(c_double), allocatable :: lmax(:)
  integer(c_int64_t), allocatable :: lidx(:)

  call c_f_pointer(a_ptr, a, shape=[int(len_1d)])
  call c_f_pointer(result_ptr, result, shape=[1])

  if (inc == 1) then
    nt = omp_get_max_threads()
    if (nt < 1) nt = 1
    allocate(lmax(nt), lidx(nt))
    lmax = -1.0d0
    lidx = 0
    !$omp parallel do private(start, end_, maxv, v, i) schedule(static)
    do tid = 1, nt
       start = (len_1d * (tid - 1)) / nt
       end_  = (len_1d * tid) / nt
       maxv = -1.0d0
       do i = start, end_ - 1
          v = abs(a(1 + i))
          if (v > maxv) then
             maxv = v
             lidx(tid) = i
          end if
       end do
       lmax(tid) = maxv
    end do
    !$omp end parallel do
    gmaxv = -1.0d0
    gindex = 0
    do tid = 1, nt
       if (lmax(tid) > gmaxv) then
          gmaxv = lmax(tid)
          gindex = lidx(tid)
       end if
    end do
  else
    maxv = abs(a(1))
    gindex = 0
    do i = 1, len_1d - 1
       v = abs(a(1 + i * inc))
       if (v > maxv) then
          maxv = v
          gindex = i
       end if
    end do
    gmaxv = maxv
  end if

  result(1) = gmaxv + real(gindex, c_double)
end subroutine
